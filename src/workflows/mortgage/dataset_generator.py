"""Generate synthetic mortgage profiles and render PDFs/images."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from google import genai
from google.genai import types

from .mortgage_models import MortgageApplication
from .mortgage_utils import compute_metrics, determine_decision, parse_llm_json

DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_BATCH_SIZE = 5
DEFAULT_COUNT = 50
DEFAULT_OUTPUT_DIR = "datasets"
DEFAULT_DPI = 200


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_dotenv() -> None:
    """Load .env values into the process environment (non-destructive)."""

    env_path = _repo_root() / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'").strip()
        if key and key not in os.environ:
            os.environ[key] = value


def _gemini_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment.")
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )


def _load_seed_profiles() -> list[dict[str, Any]]:
    data_path = _repo_root() / "resources" / "mortgage_test_cases.json"
    payload = json.loads(data_path.read_text())
    return payload["test_cases"]


def _merge_defaults(template: Any, data: Any) -> Any:
    if isinstance(template, dict):
        merged: dict[str, Any] = {}
        data = data or {}
        for key, value in template.items():
            if key in data:
                merged[key] = _merge_defaults(value, data[key])
            else:
                merged[key] = copy.deepcopy(value)
        for key, value in data.items():
            if key not in merged:
                merged[key] = value
        return merged
    if isinstance(template, list):
        if isinstance(data, list) and data:
            return data
        return copy.deepcopy(template)
    return data if data is not None else copy.deepcopy(template)


def _normalize_profile(raw: dict[str, Any], template: dict[str, Any], index: int) -> MortgageApplication:
    merged = _merge_defaults(template, raw)
    merged["case_id"] = f"MTG-2026-{index:03d}"

    debts = merged.get("debts", {})
    total_debt = (
        float(debts.get("car_loan", 0))
        + float(debts.get("student_loan", 0))
        + float(debts.get("credit_cards", 0))
        + float(debts.get("personal_loan", 0))
    )
    debts["total_monthly_debt"] = round(total_debt, 2)
    merged["debts"] = debts

    assets = merged.get("assets", {})
    if "liquid_assets_total" not in assets:
        assets["liquid_assets_total"] = round(
            float(assets.get("checking", 0)) + float(assets.get("savings", 0)),
            2,
        )
    merged["assets"] = assets

    loan = merged.get("loan", {})
    if "monthly_piti" not in loan:
        loan["monthly_piti"] = loan.get("estimated_payment", 0)
    merged["loan"] = loan

    monthly_income = float(merged.get("employment", {}).get("monthly_income", 0))
    dti_ratio = total_debt / monthly_income if monthly_income > 0 else 0
    merged["dti_ratio"] = round(dti_ratio, 4)

    app = MortgageApplication.model_validate(merged)
    metrics = compute_metrics(app)
    decision = determine_decision(app, metrics)
    return app.model_copy(update={"expected_decision": decision, "dti_ratio": metrics.dti_ratio})


def _format_currency(value: Any) -> str:
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return str(value)


def _render_pdf(profile: MortgageApplication, output_path: Path) -> None:
    doc = fitz.open()
    page = doc.new_page()
    width, height = page.rect.width, page.rect.height
    x = 40
    y = 40
    line_height = 14

    def new_page() -> None:
        nonlocal page, y
        page = doc.new_page()
        y = 40

    def add_line(text: str, bold: bool = False, size: int = 10) -> None:
        nonlocal y
        if y > height - 40:
            new_page()
        page.insert_text((x, y), text, fontsize=size, fontname="helv")
        y += line_height

    add_line("Mortgage Application Summary", bold=True, size=14)
    y += 6

    add_line("Applicant Information", bold=True, size=12)
    add_line(f"Case ID: {profile.case_id}")
    add_line(f"Name: {profile.name}")
    add_line(f"SSN: {profile.ssn}")
    add_line(f"Email: {profile.email}")
    add_line(f"Phone: {profile.phone}")
    add_line(f"Address: {profile.address}")
    add_line(f"Credit Score: {profile.credit_score}")
    add_line(f"DTI Ratio: {profile.dti_ratio:.2%}")
    add_line(f"Expected Decision: {profile.expected_decision}")
    y += 6

    add_line("Employment & Income", bold=True, size=12)
    emp = profile.employment
    add_line(f"Employer: {emp.employer}")
    add_line(f"Position: {emp.position}")
    add_line(f"Years: {emp.years}")
    add_line(f"Monthly Income: {_format_currency(emp.monthly_income)}")
    add_line(f"Type: {emp.type}")
    add_line(f"Employment Gap: {emp.employment_gap}")
    add_line(f"Gap Explanation: {emp.gap_explanation}")
    add_line(f"Base Salary: {_format_currency(emp.income_details.base_salary)}")
    add_line(f"Bonus 2023: {_format_currency(emp.income_details.bonus_2023)}")
    add_line(f"Bonus 2024: {_format_currency(emp.income_details.bonus_2024)}")
    add_line(f"Bonus Stable: {emp.income_details.bonus_stable}")
    add_line(f"Employer Confirmation: {emp.income_details.employer_confirmation}")
    y += 6

    add_line("Credit History", bold=True, size=12)
    credit = profile.credit_history
    add_line(f"Bankruptcies: {credit.bankruptcies}")
    add_line(f"Foreclosures: {credit.foreclosures}")
    add_line(f"Late Payments (12 mo): {credit.late_payments_12mo}")
    add_line(f"Late Payments (24 mo): {credit.late_payments_24mo}")
    add_line(f"Collections Count: {len(credit.collections)}")
    add_line(f"Inquiries (6 mo): {credit.inquiries_6mo}")
    add_line(f"Oldest Tradeline Years: {credit.oldest_tradeline_years}")
    add_line(f"Total Tradelines: {credit.total_tradelines}")
    add_line(f"Credit Notes: {credit.credit_notes}")
    y += 6

    add_line("Debts", bold=True, size=12)
    debts = profile.debts
    add_line(f"Car Loan: {_format_currency(debts.car_loan)}")
    add_line(f"Student Loan: {_format_currency(debts.student_loan)}")
    add_line(f"Credit Cards: {_format_currency(debts.credit_cards)}")
    add_line(f"Personal Loan: {_format_currency(debts.personal_loan)}")
    add_line(f"Total Monthly Debt: {_format_currency(debts.total_monthly_debt)}")
    y += 6

    add_line("Assets", bold=True, size=12)
    assets = profile.assets
    add_line(f"Checking: {_format_currency(assets.checking)}")
    add_line(f"Savings: {_format_currency(assets.savings)}")
    add_line(f"Liquid Assets Total: {_format_currency(assets.liquid_assets_total)}")
    add_line(f"401k: {_format_currency(assets.retirement_401k)}")
    add_line(f"Reserves Months: {assets.reserves_months}")
    add_line(f"Deposit Explanations: {assets.deposit_explanations}")
    if assets.recent_deposits:
        add_line("Recent Deposits:")
        for deposit in assets.recent_deposits:
            add_line(f"- {deposit.date}: {_format_currency(deposit.amount)} ({deposit.description})")
    y += 6

    add_line("Loan Details", bold=True, size=12)
    loan = profile.loan
    add_line(f"Amount: {_format_currency(loan.amount)}")
    add_line(f"Down Payment: {_format_currency(loan.down_payment)}")
    add_line(f"Closing Costs: {_format_currency(loan.closing_costs)}")
    add_line(f"Estimated Payment: {_format_currency(loan.estimated_payment)}")
    add_line(f"Monthly PITI: {_format_currency(loan.monthly_piti)}")
    add_line(f"Property Type: {loan.property_type}")
    add_line(f"Use: {loan.use}")
    y += 6

    add_line("Property", bold=True, size=12)
    prop = profile.property
    add_line(f"Purchase Price: {_format_currency(prop.purchase_price)}")
    add_line(f"Appraised Value: {_format_currency(prop.appraised_value)}")
    add_line(f"Condition: {prop.condition}")
    add_line(f"Type: {prop.type}")
    add_line(f"Required Repairs: {_format_currency(prop.required_repairs)}")
    add_line(f"Repair Details: {prop.repair_details}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output_path)
    doc.close()


def _render_images(pdf_path: Path, output_dir: Path, dpi: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    scale = dpi / 72
    doc = fitz.open(pdf_path)
    matrix = fitz.Matrix(scale, scale)

    for index in range(doc.page_count):
        page = doc.load_page(index)
        pix = page.get_pixmap(matrix=matrix)
        image_path = output_dir / f"{pdf_path.stem}_p{index + 1}.png"
        pix.save(image_path)

    doc.close()


async def _generate_profiles(count: int, batch_size: int, model: str) -> list[MortgageApplication]:
    seeds = _load_seed_profiles()
    template = seeds[0]
    client = _gemini_client()

    profiles: list[MortgageApplication] = []
    attempts = 0
    max_attempts = count * 3

    while len(profiles) < count and attempts < max_attempts:
        remaining = count - len(profiles)
        current_batch = min(batch_size, remaining)
        attempts += 1

        prompt = f"""
You are generating synthetic mortgage application profiles.
Return JSON with a top-level key "profiles" containing {current_batch} profiles.

Rules:
- Create new realistic names, emails, phone numbers, and addresses.
- Do not reuse any names or addresses from the seed profiles.
- Keep structure and fields identical to the seed schema.
- Use plausible ranges for income, debts, assets, credit scores, and property values.
- Ensure all nested objects are present.
- Output JSON only.

Seed profiles (schema examples):
{json.dumps(seeds, indent=2)}
""".strip()

        response = await client.aio.models.generate_content(
            model=model,
            contents=prompt,
        )
        data = parse_llm_json(response.text or "")
        batch = data.get("profiles", []) if isinstance(data, dict) else []

        for item in batch:
            if not isinstance(item, dict):
                continue
            try:
                profile = _normalize_profile(item, template, len(profiles) + 1)
            except Exception:
                continue
            profiles.append(profile)
            if len(profiles) >= count:
                break

    if len(profiles) < count:
        raise RuntimeError(f"Only generated {len(profiles)} valid profiles")

    return profiles


def _write_profiles(profiles: list[MortgageApplication], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [profile.model_dump(by_alias=True) for profile in profiles]
    output_path.write_text(json.dumps(payload, indent=2))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate mortgage profile dataset")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    _load_dotenv()

    profiles = asyncio.run(_generate_profiles(args.count, args.batch_size, args.model))

    output_root = Path(args.output)
    pdf_dir = output_root / "pdfs"
    image_dir = output_root / "images"

    for profile in profiles:
        pdf_path = pdf_dir / f"{profile.case_id}.pdf"
        _render_pdf(profile, pdf_path)
        _render_images(pdf_path, image_dir, args.dpi)

    _write_profiles(profiles, output_root / "profiles.json")
    print(f"Generated {len(profiles)} profiles in {output_root}")


if __name__ == "__main__":
    main()
