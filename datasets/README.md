# Datasets

Synthetic inputs used by the demos and tests live here.

**Contents**
- `images/` contains OCR-ready page images named like `CASEID_p1.png`.
- `pdfs/` contains the original synthetic PDFs used to generate images.
- `uploads/` is created by the review UI when you upload new cases (ignored by git).
- `profiles.json` stores the raw synthetic profile data used for generation.

**How to use**
- Point `--image-dir` to `datasets/images` when running the demo CLI.
- The review UI writes uploaded cases into `datasets/uploads/<case_id>/`.
- All data is synthetic and intended for demos only.
