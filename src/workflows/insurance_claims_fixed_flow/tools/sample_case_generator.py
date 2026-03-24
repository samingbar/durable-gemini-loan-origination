# ruff: noqa: T201
"""Generate sample insurance claim PDFs and OCR image pages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import fitz

from src.workflows.insurance_claims_fixed_flow.insurance_models import InsuranceClaim

DEFAULT_OUTPUT_DIR = "datasets/insurance_claims"
DEFAULT_DPI = 200
PDF_SUBDIR = "pdfs"
IMAGE_SUBDIR = "images"
PROFILE_PATH = "profiles.json"
PAGE_HEIGHT = 792
PAGE_WIDTH = 612
LEFT_MARGIN = 40
TOP_MARGIN = 40
BOTTOM_MARGIN = 40
BODY_FONT_SIZE = 10
BODY_LINE_HEIGHT = 14
HEADING_FONT_SIZE = 12
HEADING_LINE_HEIGHT = 18
TITLE_FONT_SIZE = 14
TITLE_LINE_HEIGHT = 22


def _repo_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parents[4]


def _load_cases() -> list[InsuranceClaim]:
    """Load and validate the checked-in insurance claim fixtures."""
    data_path = _repo_root() / "resources" / "insurance_claim_test_cases.json"
    payload = json.loads(data_path.read_text())
    return [InsuranceClaim.model_validate(case) for case in payload["test_cases"]]


def _format_currency(value: float) -> str:
    """Format a number as a currency string."""
    return f"${float(value):,.2f}"


def _format_bool(value: bool) -> str:  # noqa: FBT001
    """Format booleans consistently for PDF text."""
    return "Yes" if value else "No"


def _render_pdf(claim: InsuranceClaim, output_path: Path) -> None:  # noqa: PLR0915
    """Render a human-readable insurance claim summary PDF."""
    doc = fitz.open()
    page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    y = TOP_MARGIN

    def new_page() -> None:
        nonlocal page, y
        page = doc.new_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
        y = TOP_MARGIN

    def add_line(
        text: str, *, size: int = BODY_FONT_SIZE, line_height: int = BODY_LINE_HEIGHT
    ) -> None:
        nonlocal y
        if y > PAGE_HEIGHT - BOTTOM_MARGIN:
            new_page()
        page.insert_text((LEFT_MARGIN, y), text, fontsize=size, fontname="helv")
        y += line_height

    def add_blank_line() -> None:
        nonlocal y
        y += 6

    add_line(
        "Insurance Claim Intake Summary",
        size=TITLE_FONT_SIZE,
        line_height=TITLE_LINE_HEIGHT,
    )
    add_line(f"Case ID: {claim.case_id}")
    add_line(f"Expected Decision: {claim.expected_decision or 'N/A'}")
    add_blank_line()

    add_line("Claimant Information", size=HEADING_FONT_SIZE, line_height=HEADING_LINE_HEIGHT)
    add_line(f"Name: {claim.name or ''}")
    add_line(f"Policy Number: {claim.policy_number or ''}")
    add_line(f"Email: {claim.email or ''}")
    add_line(f"Phone: {claim.phone or ''}")
    add_line(f"Address: {claim.address or ''}")
    add_blank_line()

    add_line("Policy Details", size=HEADING_FONT_SIZE, line_height=HEADING_LINE_HEIGHT)
    policy = claim.policy
    add_line(f"Line of Business: {policy.line_of_business}")
    add_line(f"Policy Type: {policy.policy_type}")
    add_line(f"Coverage Confirmed: {_format_bool(policy.coverage_confirmed)}")
    add_line(f"Coverage Limit: {_format_currency(policy.coverage_limit)}")
    add_line(f"Deductible: {_format_currency(policy.deductible)}")
    add_line(f"Premium Status: {policy.premium_status}")
    add_line(f"Policy Status: {policy.policy_status}")
    add_line(f"Years Insured: {policy.years_insured}")
    add_line(f"Prior Claims (3y): {policy.prior_claims_3y}")
    exclusions = ", ".join(policy.exclusions_noted) or "None"
    add_line(f"Exclusions Noted: {exclusions}")
    add_blank_line()

    add_line("Incident Details", size=HEADING_FONT_SIZE, line_height=HEADING_LINE_HEIGHT)
    incident = claim.incident
    add_line(f"Date of Loss: {incident.date_of_loss}")
    add_line(f"Reported Date: {incident.reported_date}")
    add_line(f"Claim Type: {incident.claim_type}")
    add_line(f"Location: {incident.location}")
    add_line(f"Police Report Filed: {_format_bool(incident.police_report_filed)}")
    add_line(f"Weather Related: {_format_bool(incident.weather_related)}")
    add_line(f"Incident Description: {incident.description}")
    add_blank_line()

    add_line("Loss Details", size=HEADING_FONT_SIZE, line_height=HEADING_LINE_HEIGHT)
    loss = claim.loss
    add_line(f"Claimed Amount: {_format_currency(loss.claimed_amount)}")
    add_line(f"Estimated Damage: {_format_currency(loss.estimated_damage)}")
    add_line(f"Emergency Mitigation: {_format_currency(loss.emergency_mitigation)}")
    add_line(f"Depreciation Applied: {_format_currency(loss.depreciation_applied)}")
    add_line(f"Salvage Value: {_format_currency(loss.salvage_value)}")
    add_line(f"Repair Status: {loss.repair_status}")
    add_line(f"Loss Notes: {loss.loss_notes}")
    if loss.damaged_items:
        add_line("Damaged Items:")
        for item in loss.damaged_items:
            add_line(
                f"- {item.description} | {item.category} | {_format_currency(item.estimated_cost)}"
            )
    add_blank_line()

    add_line("Documents", size=HEADING_FONT_SIZE, line_height=HEADING_LINE_HEIGHT)
    documents = claim.documents
    add_line(f"Photos Received: {_format_bool(documents.photos_received)}")
    add_line(f"Repair Estimates Count: {documents.repair_estimates_count}")
    add_line(f"Receipts Count: {documents.receipts_count}")
    add_line(f"Witness Statements Count: {documents.witness_statements_count}")
    add_line(f"Proof of Ownership: {_format_bool(documents.proof_of_ownership)}")
    add_line(f"Adjuster Notes: {documents.adjuster_notes}")
    missing_documents = ", ".join(documents.missing_documents) or "None"
    add_line(f"Missing Documents: {missing_documents}")
    add_blank_line()

    add_line("Parties", size=HEADING_FONT_SIZE, line_height=HEADING_LINE_HEIGHT)
    parties = claim.parties
    add_line(f"Third Party Involved: {_format_bool(parties.third_party_involved)}")
    add_line(f"Third Party Details: {parties.third_party_details}")
    add_line(f"Injuries Reported: {_format_bool(parties.injuries_reported)}")
    add_line(f"Claimant Statement: {parties.claimant_statement}")
    add_line(f"Witness Summary: {parties.witness_summary}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output_path)
    doc.close()


def _render_images(pdf_path: Path, output_dir: Path, dpi: int) -> None:
    """Render a PDF into deterministic page images."""
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


def generate_sample_cases(output_dir: Path, dpi: int) -> list[InsuranceClaim]:
    """Generate PDFs and PNGs for the checked-in insurance claim fixtures."""
    claims = _load_cases()
    pdf_dir = output_dir / PDF_SUBDIR
    image_dir = output_dir / IMAGE_SUBDIR

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / PROFILE_PATH).write_text(
        json.dumps([claim.model_dump() for claim in claims], indent=2)
    )

    for claim in claims:
        pdf_path = pdf_dir / f"{claim.case_id}.pdf"
        _render_pdf(claim, pdf_path)
        _render_images(pdf_path, image_dir, dpi)

    return claims


def main() -> None:
    """Generate sample insurance claim PDFs and OCR image packs."""
    parser = argparse.ArgumentParser(description="Generate insurance claim sample PDFs and images.")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for generated files (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=DEFAULT_DPI,
        help=f"Image DPI when rendering PDF pages (default: {DEFAULT_DPI}).",
    )
    args = parser.parse_args()

    output_dir = _repo_root() / args.output_dir
    claims = generate_sample_cases(output_dir, args.dpi)
    print(f"Generated {len(claims)} insurance claim sample cases in {output_dir}")


if __name__ == "__main__":
    main()
