from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from invoice_extractor import (
    Invoice,
    build_invoice_prompt,
    extract_invoice,
    extract_labelled_total,
    extract_pdf_text,
)


def test_extract_pdf_text_combines_non_empty_pages():
    first_page = Mock(extract_text=Mock(return_value="Seller: Acme"))
    empty_page = Mock(extract_text=Mock(return_value=""))
    second_page = Mock(extract_text=Mock(return_value="Total: $12.50"))

    with patch("invoice_extractor.PdfReader") as reader_class:
        reader_class.return_value.pages = [first_page, empty_page, second_page]
        assert extract_pdf_text(Path("invoice.pdf")) == "Seller: Acme\n\nTotal: $12.50"


def test_extract_pdf_text_rejects_scanned_or_empty_pdf():
    with patch("invoice_extractor.PdfReader") as reader_class:
        reader_class.return_value.pages = [Mock(extract_text=Mock(return_value=" "))]
        with pytest.raises(ValueError, match="No selectable text"):
            extract_pdf_text(Path("scanned.pdf"))


def test_prompt_distinguishes_seller_from_customer_and_total_from_line_items():
    prompt = build_invoice_prompt("SuperStore\nBill To: Zuschuss Donatelli\nTotal: $2,078.96")

    assert "not the Bill To or Ship To customer" in prompt
    assert "Prefer the field labelled Total" in prompt
    assert "Ignore line-item amounts" in prompt


def test_extract_labelled_total_uses_final_total_not_subtotal_or_line_item():
    text = """Item $4,012.08
Subtotal: $4,012.08
Shipping: $72.92
Total:
:
$2,078.96
"""

    assert extract_labelled_total(text) == 2078.96


def test_extract_labelled_total_supports_european_format():
    assert extract_labelled_total("Total: 1.234,56 EUR") == 1234.56


def test_extract_invoice_returns_typed_result():
    expected = Invoice(company="SuperStore", total=2078.96)
    with patch("invoice_extractor.extract_pdf_text", return_value="Total: $2,078.96"):
        with patch("invoice_extractor.needle.extract", return_value=expected) as extract:
            result = extract_invoice(Path("invoice.pdf"))

    assert result == expected
    extract.assert_called_once()
    assert extract.call_args.args[1] is Invoice


def test_extract_invoice_prefers_explicit_labelled_total():
    model_result = Invoice(company="SuperStore", total=20.07)
    with patch(
        "invoice_extractor.extract_pdf_text",
        return_value="SuperStore\nTotal: $2,078.96",
    ):
        with patch("invoice_extractor.needle.extract", return_value=model_result):
            result = extract_invoice(Path("invoice.pdf"))

    assert result == Invoice(company="SuperStore", total=2078.96)
