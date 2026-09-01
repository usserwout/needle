"""Extract the issuing company and final total from an invoice PDF.

Usage:
    python invoice_extractor.py /path/to/invoice.pdf

Install the optional dependencies first with:
    pip install -e '.[invoice]'
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import needle
from pydantic import BaseModel, Field
from pypdf import PdfReader


class Invoice(BaseModel):
    """The invoice fields returned by the extractor."""

    company: str = Field(description="The company that issued the invoice")
    total: float = Field(
        ge=0,
        description="The final invoice total or amount due, in major currency units",
    )


_TOTAL_LABEL = re.compile(
    r"^\s*(?:grand\s+total|invoice\s+total|total\s+due|amount\s+due|"
    r"balance\s+due|amount\s+payable|total)\s*:?(?:\s+(.*))?\s*$",
    re.IGNORECASE,
)
_AMOUNT = re.compile(
    r"(?<![\w/])(?:[$€£]\s*)?\d+(?:(?:[,.]\d{3})*[,.]\d{2}|"
    r"(?:[,.]\d{3})+)(?![\w/])"
)


def extract_pdf_text(pdf_path: Path) -> str:
    """Read and combine all text-bearing pages in an invoice PDF."""

    try:
        reader = PdfReader(str(pdf_path))
    except Exception as exc:
        raise ValueError(f"Could not read PDF '{pdf_path}': {exc}") from exc

    pages = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text.strip())

    if not pages:
        raise ValueError(
            f"No selectable text was found in '{pdf_path}'. "
            "This script does not OCR scanned invoices."
        )
    return "\n\n".join(pages)


def build_invoice_prompt(invoice_text: str) -> str:
    """Create an extraction prompt that disambiguates common invoice fields."""

    return f"""Extract the issuing company and final price from this invoice.

Rules:
- company is the seller/vendor that issued the invoice, not the Bill To or Ship To customer.
- total is the final payable invoice amount. Prefer the field labelled Total; use Balance Due only when there is no Total.
- Ignore line-item amounts, subtotal, discount, tax, and shipping amounts.
- Return total as a number in major currency units (for example, $2,078.96 becomes 2078.96).

Invoice text:
{invoice_text}
"""


def _parse_amount(value: str) -> float:
    """Parse common US and European thousands/decimal separators."""

    value = value.replace(" ", "").replace("$", "").replace("€", "").replace("£", "")
    if "," in value and "." in value:
        decimal_separator = "," if value.rfind(",") > value.rfind(".") else "."
        thousands_separator = "." if decimal_separator == "," else ","
        value = value.replace(thousands_separator, "").replace(decimal_separator, ".")
    elif "," in value:
        value = value.replace(",", "." if len(value.rsplit(",", 1)[1]) == 2 else "")
    elif value.count(".") > 1:
        value = value.replace(".", "")
    return float(value)


def extract_labelled_total(invoice_text: str) -> Optional[float]:
    """Find the last explicit Total or Balance Due amount in invoice text."""

    lines = [line.strip() for line in invoice_text.splitlines()]
    candidates = []
    for index, line in enumerate(lines):
        match = _TOTAL_LABEL.match(line)
        if not match or line.lower().startswith(("subtotal", "total tax")):
            continue
        inline_text = match.group(1) or ""
        found_amount = False
        for offset in range(0, 11):
            if index + offset >= len(lines):
                break
            amounts = _AMOUNT.findall(lines[index + offset] if offset else inline_text)
            for amount in amounts:
                candidates.append(_parse_amount(amount))
            if amounts:
                found_amount = True
                break
        if not found_amount:
            for offset in range(1, 11):
                if index - offset < 0:
                    break
                amounts = _AMOUNT.findall(lines[index - offset])
                for amount in amounts:
                    candidates.append(_parse_amount(amount))
                if amounts:
                    break
    return candidates[-1] if candidates else None


def extract_invoice(pdf_path: Path, max_new_tokens: int = 256) -> Invoice:
    """Extract the issuing company and final total from ``pdf_path``."""

    invoice_text = extract_pdf_text(pdf_path)
    invoice = needle.extract(
        build_invoice_prompt(invoice_text),
        Invoice,
        max_new_tokens=max_new_tokens,
    )
    if invoice is None:
        raise ValueError("Needle could not extract an invoice from the PDF")
    labelled_total = extract_labelled_total(invoice_text)
    if labelled_total is not None:
        invoice = Invoice(company=invoice.company, total=labelled_total)
    return invoice


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract the issuing company and final total from an invoice PDF."
    )
    parser.add_argument("pdf", type=Path, help="Path to the invoice PDF")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum model output tokens (default: 256)",
    )
    args = parser.parse_args(argv)

    invoice = extract_invoice(args.pdf, max_new_tokens=args.max_new_tokens)
    print(f"Company: {invoice.company}")
    print(f"Final price: {invoice.total:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
