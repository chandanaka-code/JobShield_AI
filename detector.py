
"""
JobShield AI — job offer fraud detection.

Combines:
- EMSCAD-trained machine-learning model (if available)
- Rule-based warning signals
- Sender email checks
- Company domain comparison
- TXT, PDF, DOCX, and image text extraction
- OCR for images and scanned PDFs

Scores are screening indicators, not definitive fraud verdicts
or calibrated real-world probabilities.
"""

from pathlib import Path
import re
import email.utils
import pytesseract

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

FREE_PROVIDERS = {
    "gmail.com",
    "yahoo.com",
    "outlook.com",
    "hotmail.com",
    "live.com",
    "icloud.com",
    "proton.me",
    "protonmail.com",
    "rediffmail.com",
}

URGENT_PHRASES = [
    ("pay registration fee", 25),
    ("registration fee", 18),
    ("processing fee", 18),
    ("security deposit", 22),
    ("pay to apply", 25),
    ("training fee", 18),
    ("guaranteed placement", 12),
    ("limited slots", 8),
    ("act immediately", 10),
    ("offer expires today", 10),
    ("send your aadhaar", 10),
    ("bank details", 12),
    ("otp", 12),
    ("whatsapp only", 8),
    ("no interview", 12),
    ("work from home", 3),
    ("earn per day", 8),
    ("refundable fee", 20),
]

URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>()]+", re.I)

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "job_fraud_model.joblib"
)
COMPANY_RE = re.compile(
    r"\b([A-Z][A-Za-z0-9&.,' -]{1,70}?"
    r"(?:Private Limited|Pvt\.?\s*Ltd\.?|"
    r"Limited|Ltd\.?|LLP|Inc\.?|Corporation|Corp\.?))\b",
    re.IGNORECASE,
)

def extract_company_name(text, supplied_name=""):
    """Find a company name in the offer text."""
    if supplied_name.strip():
        
        return supplied_name.strip()

    for line in (text or "").splitlines():
        line = line.strip()
        match = COMPANY_RE.search(line)
        if match:
            return match.group(1).strip(" .,|-")

    return ""
_ML_MODEL = None
_ML_MODEL_LOADED = False




# --------------------------------------------------
# MACHINE-LEARNING MODEL
# --------------------------------------------------

def _load_job_model():
    """Load the trained model once, if it exists."""
    global _ML_MODEL, _ML_MODEL_LOADED

    if _ML_MODEL_LOADED:
        return _ML_MODEL

    _ML_MODEL_LOADED = True

    if not MODEL_PATH.exists():
        return None

    try:
        import joblib

        _ML_MODEL = joblib.load(MODEL_PATH)

    except Exception:
        _ML_MODEL = None

    return _ML_MODEL


def _predict_job_fraud(text):
    """Return the model prediction and fraud-class score, if available."""
    model = _load_job_model()

    if model is None or not text.strip():
        return None

    try:
        prediction = int(model.predict([text])[0])
        probability = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([text])[0]

            classes = list(
                getattr(model, "classes_", [])
            )

            if not classes and hasattr(model, "named_steps"):
                classifier = model.named_steps.get("classifier")

                if classifier is not None:
                    classes = list(
                        getattr(classifier, "classes_", [])
                    )

            if 1 in classes:
                probability = (
                    float(probabilities[classes.index(1)]) * 100
                )
            elif len(probabilities) > 1:
                probability = float(probabilities[-1]) * 100

        return {
            "prediction": prediction,
            "score": probability,
        }

    except Exception:
        return None


# --------------------------------------------------
# EMAIL ANALYSIS
# --------------------------------------------------

def _domain(email_address):
    """Extract the domain from an email address."""
    parsed = email.utils.parseaddr(email_address)[1]

    if "@" not in parsed:
        return ""

    return parsed.rsplit("@", 1)[1].strip().lower().rstrip(".")


def analyze_sender_email(
    sender_email,
    company_name="",
    official_domain="",
):
    """Check an email address for potential warning signs."""
    sender_email = (sender_email or "").strip()
    parsed = email.utils.parseaddr(sender_email)[1]

    reasons = []
    advice = []
    score = 0

    if not parsed or not re.fullmatch(
        r"[^@\s]+@[^@\s]+\.[^@\s]+",
        parsed,
    ):
        reasons.append(
            "The sender address does not have a standard email format."
        )
        score += 20
        domain = ""

    else:
        domain = _domain(sender_email)

        if domain in FREE_PROVIDERS:
            reasons.append(
                f"The sender uses a public email provider ({domain}), "
                "not a company-specific domain."
            )
            score += 12

        if (
            re.search(r"\d{3,}", domain)
            or len(domain.split(".")[0]) > 28
        ):
            reasons.append(
                "The domain contains an unusual number pattern "
                "or an unusually long name."
            )
            score += 10

        recruiting_words = (
            "internship",
            "career",
            "hr",
            "recruitment",
            "jobs",
        )

        if any(word in domain for word in recruiting_words) and company_name:
            comp = re.sub(
                r"[^a-z0-9]", "", company_name.lower()
            )
            dom = re.sub(
                r"[^a-z0-9]", "", domain.split(".")[0]
            )

            if comp and comp[:6] not in dom and dom[:6] not in comp:
                reasons.append(
                    "The domain uses recruiting-related words but "
                    "does not clearly resemble the supplied company name."
                )
                score += 18

        if official_domain:
            official = (
                official_domain.lower()
                .strip()
                .replace("https://", "")
                .replace("http://", "")
                .split("/")[0]
            )

            if official.startswith("www."):
                official = official[4:]

            if (
                domain
                and domain != official
                and not domain.endswith("." + official)
            ):
                reasons.append(
                    f"Sender domain ({domain}) does not match "
                    f"the supplied official domain ({official})."
                )
                score += 35

        if not reasons:
            reasons.append(
                "No obvious address-format warning was found. "
                "This does not prove the sender is genuine."
            )

    if not official_domain:
        advice.append(
            "Compare the sender domain with the company's official "
            "website, found independently."
        )

    advice.extend([
        "Check the original email's SPF, DKIM, and DMARC results if available.",
        "Do not pay fees or share OTPs, passwords, identity documents, "
        "or bank details based only on an email.",
    ])

    score = min(100, score)

    if score >= 50:
        level = "High risk"
    elif score >= 20:
        level = "Needs review"
    else:
        level = "Lower apparent risk"

    return {
        "score": score,
        "level": level,
        "domain": domain,
        "reasons": reasons,
        "advice": advice,
    }


# --------------------------------------------------
# JOB OFFER ANALYSIS
# --------------------------------------------------

def analyze_offer(text):
    """Analyze extracted or manually entered job-offer text."""
    text = (text or "").strip()
    lower = text.lower()

    heuristic_score = 0
    reasons = []
    signals = []
    advice = []

    if not text:
        return {
            "score": 0,
            "level": "Not assessed",
            "label": "No message text supplied",
            "reasons": [],
            "signals": [],
            "advice": ["Add offer text for content analysis."],
            "ml_prediction": None,
            "ml_score": None,
        }

    for phrase, weight in URGENT_PHRASES:
        if phrase in lower:
            heuristic_score += weight
            reasons.append(f"Contains the phrase “{phrase}”.")
            signals.append(phrase)

    urls = URL_RE.findall(text)

    if len(urls) >= 3:
        heuristic_score += 8
        reasons.append(
            "Contains several links; inspect each destination carefully."
        )

    if (
        re.search(r"\b(?:pay|transfer|deposit|fee)\b", lower)
        and re.search(
            r"\b(?:intern|job|offer|placement|recruit)\w*\b",
            lower,
        )
    ):
        heuristic_score += 10
        reasons.append(
            "Payment-related wording appears in an employment context."
        )

    if re.search(
        r"\b(?:aadhaar|pan card|passport|bank account|account number|otp|password)\b",
        lower,
    ):
        heuristic_score += 8
        reasons.append(
            "Requests or mentions sensitive personal or financial information."
        )

    heuristic_score = min(100, heuristic_score)
    ml = _predict_job_fraud(text)

    if ml is not None and ml["score"] is not None:
        score = round(
            (heuristic_score + ml["score"]) / 2
        )

        ml_label = (
            "Fraudulent-like"
            if ml["prediction"] == 1
            else "Genuine-like"
        )

        reasons.insert(
            0,
            f"EMSCAD-trained model classified this text as {ml_label} "
            f"(model score: {ml['score']:.1f}/100; "
            "not a calibrated probability).",
        )

        label = "Machine-learning + rule-based screening"

    elif ml is not None:
        score = max(
            heuristic_score,
            100 if ml["prediction"] == 1 else 0,
        )

        reasons.insert(
            0,
            "The trained EMSCAD model was used; "
            "probability scoring is unavailable.",
        )

        label = "Machine-learning + rule-based screening"

    else:
        score = heuristic_score
        label = "Rule-based warning-signal screening"

    score = max(0, min(100, int(score)))

    if score >= 60:
        level = "High risk"
    elif score >= 30:
        level = "Needs review"
    else:
        level = "Lower apparent risk"

    if not reasons:
        reasons.append(
            "No configured high-risk phrases were detected. "
            "This does not establish that the offer is legitimate."
        )

    advice = [
        "Verify the vacancy on the company's official careers page, "
        "reached independently.",
        "Never pay an application, interview, equipment, training, "
        "or refundable security fee without independent verification.",
        "Treat this score as a screening aid, not a probability or final verdict.",
    ]

    if ml is None:
        advice.append(
            "The trained model was not loaded; this result uses "
            "rule-based checks only."
        )

    return {
        "score": score,
        "level": level,
        "label": label,
        "reasons": reasons,
        "signals": signals,
        "advice": advice,
        "ml_prediction": (
            ml["prediction"] if ml is not None else None
        ),
        "ml_score": (
            round(ml["score"], 1)
            if ml is not None and ml["score"] is not None
            else None
        ),
    }


# --------------------------------------------------
# COMPANY DOMAIN VERIFICATION
# --------------------------------------------------

def verify_company_domain(
    company_name="",
    official_domain="",
    sender_email="",
):
    """Compare the sender domain with a user-supplied official domain."""
    sender_domain = _domain(sender_email)

    official = (
        (official_domain or "")
        .strip()
        .lower()
        .replace("https://", "")
        .replace("http://", "")
        .split("/")[0]
    )

    if official.startswith("www."):
        official = official[4:]

    if not company_name and not official and not sender_email:
        return {
            "status": "Insufficient information",
            "message": (
                "Enter a company name, official domain, or sender email."
            ),
        }

    if sender_domain and official:
        match = (
            sender_domain == official
            or sender_domain.endswith("." + official)
        )

        return {
            "status": (
                "Domain matches supplied value"
                if match
                else "Domain mismatch"
            ),
            "match": match,
            "message": (
                "The sender domain matches the official domain you entered. "
                "This is not proof that the email is authentic."
                if match
                else
                "The sender domain does not match the official domain "
                "you entered. Confirm the official domain independently."
            ),
        }

    return {
        "status": "Manual verification required",
        "match": None,
        "message": (
            "This tool does not query a live company registry or prove "
            "company ownership. Find the official website independently "
            "and compare domains."
        ),
    }


# --------------------------------------------------
# FILE TEXT EXTRACTION + OCR
# --------------------------------------------------

def extract_text_from_file(path):
    """
    Extract text from TXT, DOCX, and PDF files.
    Use Tesseract OCR for images and scanned/image-only PDFs.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    # Plain text
    if suffix == ".txt":
        return path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

    # Word documents
    if suffix == ".docx":
        try:
            from docx import Document
        except ImportError as exc:
            raise RuntimeError(
                "DOCX support requires python-docx. Run: "
                "python -m pip install python-docx"
            ) from exc

        document = Document(str(path))

        return "\n".join(
            paragraph.text
            for paragraph in document.paragraphs
        )

    # PDF: try embedded text first, then OCR scanned pages
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError(
                "PDF support requires pypdf. Run: "
                "python -m pip install pypdf"
            ) from exc

        text = "\n".join(
            page.extract_text() or ""
            for page in PdfReader(str(path)).pages
        ).strip()

        if text:
            return text

        # No embedded text: OCR the rendered PDF pages.
        try:
            import fitz
            import pytesseract
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "OCR for scanned PDFs requires PyMuPDF, Pillow, "
                "and pytesseract. Run: "
                "python -m pip install PyMuPDF Pillow pytesseract"
            ) from exc

        extracted_pages = []

        with fitz.open(str(path)) as pdf:
            for page in pdf:
                pix = page.get_pixmap(
                    matrix=fitz.Matrix(2, 2),
                    alpha=False,
                )

                image = Image.frombytes(
                    "RGB",
                    (pix.width, pix.height),
                    pix.samples,
                )

                extracted_pages.append(
                    pytesseract.image_to_string(image)
                )

        return "\n".join(extracted_pages).strip()

    # Image files
    if suffix in {
        ".png", ".jpg", ".jpeg",
        ".bmp", ".tif", ".tiff",
    }:
        try:
            from PIL import Image
            import pytesseract
        except ImportError as exc:
            raise RuntimeError(
                "Image OCR requires Pillow and pytesseract. Run: "
                "python -m pip install Pillow pytesseract"
            ) from exc

        with Image.open(path) as image:
            return pytesseract.image_to_string(image)

    raise RuntimeError(
        "Unsupported file type. Upload a TXT, PDF, DOCX, PNG, "
        "JPG, JPEG, BMP, TIF, or TIFF file."
    )


# --------------------------------------------------
# DEMONSTRATION METRICS
# --------------------------------------------------

def evaluate_demo_rules():
    """
    Calculate metrics on toy examples for demonstration only.
    These examples are not a validated model benchmark.
    """
    examples = [
        ("Please pay a registration fee to secure your internship", 1),
        ("Congratulations, send OTP and bank details to process your offer", 1),
        ("Your interview is scheduled for Tuesday; see the official careers portal", 0),
        ("Please review the attached job description and apply through our company site", 0),
        ("Pay a refundable security deposit today for guaranteed placement", 1),
        ("Thank you for applying. Our recruiter will contact you after review", 0),
    ]

    predictions = [
        1 if analyze_offer(text)["score"] >= 30 else 0
        for text, _ in examples
    ]

    truth = [label for _, label in examples]

    tp = sum(
        p == 1 and y == 1
        for p, y in zip(predictions, truth)
    )
    tn = sum(
        p == 0 and y == 0
        for p, y in zip(predictions, truth)
    )
    fp = sum(
        p == 1 and y == 0
        for p, y in zip(predictions, truth)
    )
    fn = sum(
        p == 0 and y == 1
        for p, y in zip(predictions, truth)
    )

    n = len(examples)

    return {
        "n": n,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": round((tp + tn) / n, 3),
        "precision": round(tp / (tp + fp), 3) if tp + fp else 0,
        "recall": round(tp / (tp + fn), 3) if tp + fn else 0,
        "f1": (
            round(
                2 * tp / (2 * tp + fp + fn),
                3,
            )
            if 2 * tp + fp + fn
            else 0
        ),
        "note": (
            "Toy demonstration examples only. These figures are not a "
            "validated benchmark and do not measure real-world performance."
        ),
    }
