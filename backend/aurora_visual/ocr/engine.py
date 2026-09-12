import shutil

import pytesseract
from app.models.contract import OCRItem, Warning
from pytesseract import Output


def capability(language="ind+eng"):
    if not shutil.which("tesseract"):
        return {"status": "unconfigured", "message": "Tesseract belum terpasang"}
    try:
        langs = pytesseract.get_languages()
        missing = set(language.split("+")) - set(langs)
        return {"status": "unconfigured" if missing else "ok", "message": ",".join(sorted(missing)) or None}
    except (RuntimeError, OSError):
        return {"status": "unconfigured", "message": "Tesseract tidak dapat dijalankan"}


def recognize(image, language="ind+eng"):
    cap = capability(language)
    if cap["status"] != "ok":
        return [], [Warning(code="OCR_UNAVAILABLE", message=cap["message"], component="ocr")]
    try:
        data = pytesseract.image_to_data(
            image, lang=language, config="--psm 11", output_type=Output.DICT, timeout=20
        )
        items = []
        for i, text in enumerate(data["text"]):
            if not text.strip():
                continue
            x, y, w, h = (int(data[k][i]) for k in ("left", "top", "width", "height"))
            if w <= 0 or h <= 0:
                continue
            confidence = float(data["conf"][i])
            items.append(
                OCRItem(
                    text=text,
                    bbox=(
                        x / image.width,
                        y / image.height,
                        min(1.0, (x + w) / image.width),
                        min(1.0, (y + h) / image.height),
                    ),
                    confidence=confidence / 100 if 0 <= confidence <= 100 else None,
                    language=language,
                )
            )
        warnings = [
            Warning(
                code="OCR_OBSERVATION_ONLY",
                message="OCR merekam tulisan, bukan waktu pengambilan foto atau kebenaran isi tulisan.",
                component="ocr",
            )
        ]
        return items, warnings
    except (RuntimeError, OSError, pytesseract.TesseractError):
        return [], [
            Warning(code="OCR_FAILED", message="OCR gagal atau melewati batas waktu.", component="ocr")
        ]
