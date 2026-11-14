# JobShield AI — Complete Starter Project

**Project title:** AI-Based Detection of Fake Internship Offers Using Job Descriptions and Communication Patterns

This Flask project includes six modules:
1. Professional dashboard
2. Multi-input analysis (text, TXT, PDF, DOCX, image OCR, and EML)
3. Explainable risk report
4. Company/domain comparison
5. Evaluation demonstration and history
6. Sender email warning-signal analysis

## Important model note
This starter package uses **transparent rule-based screening**, not a trained machine-learning model. Scores are heuristic risk indicators, not calibrated probabilities. The evaluation page uses six toy examples only and explicitly labels them as a demonstration, not validated performance. If you have an existing trained JobShield model, integrate it into `detector.py` after reviewing its expected input schema and training labels.

## Windows setup
1. Install Python 3.11 or 3.12 for best package compatibility (Python 3.14 may not yet be supported by every dependency).
2. Extract the ZIP.
3. Open the extracted `JobShield_AI_Complete` folder in VS Code.
4. Open Terminal → New Terminal.
5. Run:
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   python app.py
   ```
6. Open `http://127.0.0.1:5000`.

If PowerShell blocks activation, use Command Prompt terminal:
```bat
.venv\Scripts\activate.bat
```

## OCR setup (optional)
Image OCR requires Tesseract OCR installed separately on Windows. Install it from the official UB Mannheim Tesseract builds. If installed to `C:\Program Files\Tesseract-OCR\tesseract.exe`, add this after `import pytesseract` in the image branch of `detector.py`:
```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```
If OCR is unavailable, paste the text manually or upload a text-based PDF/DOCX/TXT.

## Privacy and safety
Uploads are processed locally by this Flask app and deleted after extraction. Analysis history is saved in `data/history.json` on your computer. This is a development app; do not expose it publicly without authentication, CSRF protection, upload limits, secure configuration, and a production server. Email address/domain checks do not authenticate the sender. To check SPF/DKIM/DMARC, inspect the original message headers in your email provider.
