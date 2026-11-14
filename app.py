
from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from pathlib import Path
from datetime import datetime
from email import policy
from email.parser import BytesParser
from urllib.parse import quote_plus
import os
import json
import re

from detector import (
    analyze_offer,
    extract_text_from_file,
    analyze_sender_email,
    verify_company_domain,
)

app = Flask(__name__)
app.secret_key = os.environ.get(
    "JOBSHIELD_SECRET", "change-this-development-key"
)

ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT / "uploads"
DATA_DIR = ROOT / "data"

UPLOAD_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

HISTORY_FILE = DATA_DIR / "history.json"

ALLOWED_EXTENSIONS = {
    ".txt", ".pdf", ".png", ".jpg",
    ".jpeg", ".eml", ".docx"
}


def load_history():
    try:
        return json.loads(
            HISTORY_FILE.read_text(encoding="utf-8")
        )
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_history(rows):
    HISTORY_FILE.write_text(
        json.dumps(rows[:100], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def extract_company_name(text, supplied_name=""):
    """
    Prefer the company name entered by the user.
    Otherwise, look for a company name ending in a common
    legal suffix, such as Pvt Ltd, Limited, LLP, or Inc.
    """
    supplied_name = (supplied_name or "").strip()

    if supplied_name:
        return supplied_name

    patterns = [
        re.compile(
            r"\b([A-Z][A-Za-z0-9&'.,() -]{0,65}?"
            r"(?:Private\s+Limited|Pvt\.?\s*Ltd\.?|"
            r"Limited|Ltd\.?|LLP|Incorporated|Inc\.?|"
            r"Corporation|Corp\.?))\b",
            re.IGNORECASE,
        )
    ]

    for line in (text or "").splitlines():
        line = line.strip()

        # Avoid extracting very long paragraphs as company names.
        if not line or len(line) > 180:
            continue

        for pattern in patterns:
            match = pattern.search(line)

            if match:
                name = match.group(1).strip(" .,|-")
                if name:
                    return name

    return ""


def make_company_search_url(company):
    """Create a search link, not a claim of official verification."""
    if not company:
        return ""

    return (
        "https://www.google.com/search?q="
        + quote_plus(company + " official website")
    )


@app.route("/", methods=["GET"])
def index():
    history = load_history()
    return render_template(
        "index.html",
        history=history[:5],
        total=len(history),
    )


@app.route("/analyze", methods=["POST"])
def analyze():
    text = (request.form.get("offer_text") or "").strip()
    sender = (request.form.get("sender_email") or "").strip()
    company = (request.form.get("company_name") or "").strip()
    official_domain = (
        request.form.get("official_domain") or ""
    ).strip()

    uploaded = request.files.get("offer_file")
    file_name = ""

    if uploaded and uploaded.filename:
        safe_name = secure_filename(uploaded.filename)

        if not safe_name:
            flash("Please choose a valid filename.", "error")
            return redirect(url_for("index"))

        suffix = Path(safe_name).suffix.lower()

        if suffix not in ALLOWED_EXTENSIONS:
            flash(
                "Unsupported file type. Use TXT, PDF, PNG, JPG, DOCX, or EML.",
                "error",
            )
            return redirect(url_for("index"))

        # Use a unique temporary filename to avoid overwriting
        # another upload with the same name.
        target = UPLOAD_DIR / (
            datetime.now().strftime("%Y%m%d%H%M%S%f")
            + "_" + safe_name
        )

        uploaded.save(target)
        file_name = safe_name

        try:
            if suffix == ".eml":
                msg = BytesParser(
                    policy=policy.default
                ).parsebytes(target.read_bytes())

                sender = sender or str(msg.get("From", ""))
                body = msg.get_body(preferencelist=("plain",))

                if body:
                    text = (
                        text + "\n\n" + body.get_content()
                    ).strip()
                elif msg.get_content_type() == "text/plain":
                    text = (
                        text + "\n\n" + msg.get_content()
                    ).strip()
            else:
                extracted = extract_text_from_file(target)
                text = (text + "\n\n" + extracted).strip()

        except Exception:
            app.logger.exception("Could not read uploaded file")
            flash(
                "Could not read the uploaded file. "
                "Check the file format and try again.",
                "error",
            )
            return redirect(url_for("index"))

        finally:
            try:
                target.unlink()
            except OSError:
                pass

    if not text and not sender:
        flash(
            "Enter offer text, a sender email, or upload a file.",
            "error",
        )
        return redirect(url_for("index"))

    # Detect company name from the poster/text if not supplied.
    company = extract_company_name(text, company)

    if text:
        content_report = analyze_offer(text)
    else:
        content_report = {
            "score": 0,
            "level": "Not assessed",
            "label": "No message text supplied",
            "reasons": [],
            "signals": [],
            "advice": [
                "Add the offer text for content analysis."
            ],
            "ml_prediction": None,
            "ml_score": None,
        }

    if sender:
        sender_report = analyze_sender_email(
            sender, company, official_domain
        )
    else:
        sender_report = {
            "score": 0,
            "level": "Not assessed",
            "reasons": [],
            "domain": "",
            "advice": [
                "Add the sender email address to assess sender-related warning signs."
            ],
        }

    # A screening score, NOT a calibrated probability of fraud.
    scores = [
        report["score"]
        for report in (content_report, sender_report)
        if report.get("level") != "Not assessed"
    ]

    combined = round(sum(scores) / len(scores)) if scores else 0

    if combined >= 60:
        overall = "High risk"
    elif combined >= 30:
        overall = "Needs review"
    else:
        overall = "Lower risk"

    company_report = verify_company_domain(
        company, official_domain, sender
    )

    company_search_url = make_company_search_url(company)

    record = {
        "id": datetime.now().strftime("%Y%m%d%H%M%S%f"),
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "company": company or "Not detected",
        "company_search_url": company_search_url,
        "sender": sender or "Not provided",
        "file": file_name,
        "score": combined,
        "overall": overall,
        "content": content_report,
        "sender_check": sender_report,
        "company_check": company_report,
        "text_preview": text[:240],
    }

    history = load_history()
    history.insert(0, record)
    save_history(history)

    return render_template("report.html", report=record)


@app.route("/history")
def history():
    return render_template(
        "history.html",
        history=load_history(),
    )


@app.route("/evaluation")
def evaluation():
    from detector import evaluate_demo_rules

    metrics = evaluate_demo_rules()
    return render_template(
        "evaluation.html",
        metrics=metrics,
    )


@app.route("/company-check", methods=["GET", "POST"])
def company_check():
    result = None

    if request.method == "POST":
        result = verify_company_domain(
            request.form.get("company_name", "").strip(),
            request.form.get("official_domain", "").strip(),
            request.form.get("sender_email", "").strip(),
        )

    return render_template("company.html", result=result)


@app.route("/clear-history", methods=["POST"])
def clear_history():
    save_history([])
    flash("Analysis history cleared.", "success")
    return redirect(url_for("history"))


if __name__ == "__main__":
    app.run(debug=True)
