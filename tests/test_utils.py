"""Unit tests for the `utils` module."""

from copy import deepcopy

import pytest

from ragger.utils import format_answer, is_link


class TestFormatAnswer:
    """Tests for the `format_answer` function."""

    @pytest.mark.parametrize(
        argnames=["answer", "no_documents_reply", "expected"],
        argvalues=[
            ("answer", "no documents", "no documents"),
            ("answer", "", ""),
            ("", "", ""),
        ],
        ids=["with_no_documents_reply", "empty_no_documents_reply", "empty_answer"],
    )
    def test_no_documents(self, answer, no_documents_reply, expected):
        """Test that the function returns the correct answer."""
        formatted = format_answer(
            answer=answer, documents=[], no_documents_reply=no_documents_reply
        )
        assert formatted == expected

    @pytest.mark.parametrize(
        argnames=["answer", "expected"],
        argvalues=[
            (
                "answer",
                "answer<br><br>Kilde:<br><details><summary>{id}</summary>{text}</details>",
            ),
            ("", "<br><br>Kilde:<br><details><summary>{id}</summary>{text}</details>"),
        ],
        ids=["with_answer", "without_answer"],
    )
    def test_single_document(self, answer, documents, expected):
        """Test that the answer is formatted correctly with a single document."""
        formatted = format_answer(
            answer=answer, documents=documents[:1], no_documents_reply="no documents"
        )
        assert formatted == expected.format(id=documents[0].id, text=documents[0].text)

    @pytest.mark.parametrize(
        argnames=["answer", "expected"],
        argvalues=[
            (
                "answer",
                "answer<br><br>Kilder:<br>"
                "<details><summary>{first_id}</summary>{first_text}</details><br>"
                "<details><summary>{second_id}</summary>{second_text}</details>",
            ),
            (
                "",
                "<br><br>Kilder:<br>"
                "<details><summary>{first_id}</summary>{first_text}</details><br>"
                "<details><summary>{second_id}</summary>{second_text}</details>",
            ),
        ],
        ids=["with_answer", "without_answer"],
    )
    def test_multiple_documents(self, answer, documents, expected):
        """Test that the answer is formatted correctly with multiple documents."""
        formatted = format_answer(
            answer=answer, documents=documents[:2], no_documents_reply="no documents"
        )
        assert formatted == expected.format(
            first_id=documents[0].id,
            first_text=documents[0].text,
            second_id=documents[1].id,
            second_text=documents[1].text,
        )

    @pytest.mark.parametrize(
        argnames=["document_id", "expected"],
        argvalues=[
            (
                "https://example.com",
                "answer<br><br>Kilde:<br>"
                "<details><summary><a href='{id}'>{id}</a></summary>{text}</details>",
            ),
            (
                "https://www.example.com",
                "answer<br><br>Kilde:<br>"
                "<details><summary><a href='{id}'>{id}</a></summary>{text}</details>",
            ),
        ],
        ids=["with_answer", "without_answer"],
    )
    def test_document_with_link_id(self, document_id, documents, expected):
        """Test that the answer is formatted correctly with a single document."""
        document = deepcopy(x=documents[0])
        document.id = document_id
        formatted = format_answer(
            answer="answer", documents=[document], no_documents_reply="no documents"
        )
        assert formatted == expected.format(id=document.id, text=document.text)


@pytest.mark.parametrize(
    argnames=["text", "expected"],
    argvalues=[
        ("https://example.com", True),
        ("http://example.com", True),
        ("www.example.com", True),
        ("example.com", True),
        ("example", False),
        ("", False),
        ("this is not a link.com", False),
    ],
    ids=["https", "http", "www", "no_www", "no_link", "empty", "no_link_text"],
)
def test_is_link(text, expected):
    """Test that the function correctly identifies links."""
    assert is_link(text=text) == expected
