from unittest.mock import patch

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse

from .assistant_tools import (
    calculate,
    get_current_datetime,
    search_my_pdfs,
)
from .evaluation import (
    EVALUATION_DOCUMENT_NAME,
    EVALUATION_DOCUMENT_TEXT,
    citation_groundedness,
    extract_tool_names,
    percentile_95,
    run_evaluation,
)
from .practice_view import build_connection_string
from .models import Document, DocumentChunk
from .services import (
    DocumentProcessingError,
    answer_document_question,
    extract_pdf_chunks,
    find_document_chunks,
)


class FakePage:
    def __init__(self, text):
        self.text = text

    def extract_text(self):
        return self.text


class FakeReader:
    is_encrypted = False

    def __init__(self, _uploaded_file):
        self.pages = [FakePage("The report says revenue increased by 20 percent.")]


class DocumentAssistantTests(TestCase):
    def test_evaluation_tool_selection_and_latency_metrics(self):
        messages = [
            {"tool_calls": [{"name": "calculate"}]},
            {"content": "The answer is 42."},
        ]
        self.assertEqual(extract_tool_names(messages), ["calculate"])
        self.assertEqual(percentile_95([1, 2, 3, 4, 5]), 5)
        self.assertEqual(percentile_95([]), 0)

    def test_citation_groundedness_requires_valid_source_and_supported_terms(self):
        evidence = {
            (EVALUATION_DOCUMENT_NAME, 1): EVALUATION_DOCUMENT_TEXT,
        }

        supported_score = citation_groundedness(
            "Project Juniper reduced report preparation time by 40 percent "
            "[evaluation-notes.pdf, p. 1].",
            evidence,
        )
        unsupported_score = citation_groundedness(
            "Unrelated unsupported claim [other.pdf, p. 9].",
            evidence,
        )
        uncited_score = citation_groundedness(
            "Project Juniper reduced report preparation time by 40 percent.",
            evidence,
        )

        self.assertGreater(supported_score, 0.9)
        self.assertEqual(unsupported_score, 0)
        self.assertEqual(uncited_score, 0)

    def test_evaluation_harness_reports_metrics_and_cleans_up_fixture(self):
        class FakeAgent:
            def invoke(self, state, _config):
                prompt = state["messages"][-1]["content"]
                if "calculator" in prompt:
                    name, answer = "calculate", "27 * 14 = 378."
                elif "date/time" in prompt:
                    name, answer = "get_current_datetime", "Current time is 12:00 UTC."
                elif "Search my PDF" in prompt:
                    name = "search_my_pdfs"
                    answer = (
                        "Project Juniper reduced preparation time by 40 percent "
                        "[evaluation-notes.pdf, p. 1]."
                    )
                elif "remember that the project codename" in prompt:
                    name, answer = None, "I will remember that."
                else:
                    name, answer = None, "The project codename is Maple."

                messages = []
                if name:
                    messages.append({"tool_calls": [{"name": name}]})
                messages.append({"content": answer})
                return {"messages": messages}

        report = run_evaluation(FakeAgent(), session_key="evaluation-test")

        self.assertEqual(report["metrics"]["tool_selection_accuracy"], 1)
        self.assertGreater(report["metrics"]["citation_groundedness"], 0.9)
        self.assertEqual(report["metrics"]["multi_turn_consistency"], 1)
        self.assertGreaterEqual(report["metrics"]["p95_latency_ms"], 0)
        self.assertFalse(
            Document.objects.filter(session_key="evaluation-test").exists()
        )

    def test_connection_string_quotes_reserved_credentials(self):
        with patch.dict(
            "os.environ",
            {
                "DB_NAME": "workflow db",
                "DB_USER": "user@example.com",
                "DB_PASSWORD": "p@ss/word",
                "DB_HOST": "db",
                "DB_PORT": "5432",
            },
        ):
            connection_string = build_connection_string()

        self.assertEqual(
            connection_string,
            "postgresql://user%40example.com:p%40ss%2Fword@db:5432/workflow%20db",
        )

    def test_calculator_supports_basic_arithmetic_without_python_execution(self):
        self.assertEqual(calculate.invoke({"expression": "6 * (7 + 1)"}), "6 * (7 + 1) = 48")
        self.assertIn(
            "basic math operators",
            calculate.invoke({"expression": "__import__('os').getcwd()"}),
        )

    def test_calculator_rejects_excessive_exponents_and_division_by_zero(self):
        self.assertIn(
            "Exponents",
            calculate.invoke({"expression": "2 ** 101"}),
        )
        self.assertIn(
            "division by zero",
            calculate.invoke({"expression": "5 / 0"}),
        )

    def test_datetime_tool_returns_requested_timezone(self):
        result = get_current_datetime.invoke(
            {"timezone_name": "America/Denver"}
        )

        self.assertRegex(result, r"[+-]\d{2}:\d{2}$")

    def test_pdf_tool_requires_explicit_session_scope(self):
        with patch.dict("os.environ", {"CLI_DOCUMENT_SESSION_KEY": ""}):
            result = search_my_pdfs.invoke({"query": "project details"})

        self.assertIn("Search is disabled", result)

    def test_pdf_search_only_returns_the_configured_session_documents(self):
        own_document = Document.objects.create(
            session_key="cli-owner",
            filename="owner.pdf",
        )
        private_document = Document.objects.create(
            session_key="another-session",
            filename="private.pdf",
        )
        DocumentChunk.objects.create(
            document=own_document,
            page_number=2,
            content="The project budget is 500 dollars.",
        )
        DocumentChunk.objects.create(
            document=private_document,
            page_number=4,
            content="The project budget is 900 dollars.",
        )

        chunks = find_document_chunks("project budget", "cli-owner")

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].document.filename, "owner.pdf")
        self.assertEqual(chunks[0].page_number, 2)

        with patch.dict("os.environ", {"CLI_DOCUMENT_SESSION_KEY": "cli-owner"}):
            result = search_my_pdfs.invoke({"query": "project budget"})

        self.assertIn("[owner.pdf, page 2]", result)
        self.assertIn("500 dollars", result)
        self.assertNotIn("900 dollars", result)

    def test_question_response_is_rendered_in_the_web_page(self):
        self.client.get(reverse("app:index"))
        session_key = self.client.session.session_key
        Document.objects.create(session_key=session_key, filename="guide.pdf")

        with patch(
            "app.views.answer_document_question",
            return_value=("Use the first step [guide.pdf, p. 2].", [("guide.pdf", 2)]),
        ):
            response = self.client.post(
                reverse("app:index"),
                {
                    "action": "ask",
                    "question": "What is the first step?",
                },
            )

        self.assertContains(response, "Use the first step")
        self.assertContains(response, "guide.pdf, page 2")
        self.assertContains(response, "Delete PDF")

    def test_delete_only_removes_documents_from_current_session(self):
        self.client.get(reverse("app:index"))
        session_key = self.client.session.session_key
        own_document = Document.objects.create(
            session_key=session_key,
            filename="own.pdf",
        )
        other_document = Document.objects.create(
            session_key="another-session",
            filename="other.pdf",
        )

        response = self.client.post(
            reverse("app:index"),
            {
                "action": "delete",
                "document_id": own_document.pk,
            },
        )

        self.assertRedirects(response, reverse("app:index"))
        self.assertFalse(Document.objects.filter(pk=own_document.pk).exists())
        self.assertTrue(Document.objects.filter(pk=other_document.pk).exists())

    @patch("app.services.PdfReader", FakeReader)
    def test_upload_indexes_pdf_for_current_browser_session(self):
        response = self.client.post(
            reverse("app:index"),
            {
                "action": "upload",
                "pdf": SimpleUploadedFile(
                    "report.pdf",
                    b"%PDF-1.4 test fixture",
                    content_type="application/pdf",
                ),
            },
        )

        self.assertRedirects(response, reverse("app:index"))
        document = Document.objects.get(filename="report.pdf")
        self.assertTrue(document.session_key)
        self.assertEqual(document.chunks.count(), 1)
        self.assertEqual(document.chunks.get().page_number, 1)

    def test_non_pdf_upload_is_rejected(self):
        with self.assertRaises(DocumentProcessingError):
            extract_pdf_chunks(
                SimpleUploadedFile("notes.txt", b"not a pdf")
            )

    def test_pdf_text_is_split_into_overlapping_chunks_with_page_numbers(self):
        text = "A" * 1_500
        with patch("app.services.PdfReader") as reader_factory:
            reader_factory.return_value.is_encrypted = False
            reader_factory.return_value.pages = [FakePage(text)]
            chunks = extract_pdf_chunks(
                SimpleUploadedFile("long.pdf", b"%PDF-1.4 test")
            )

        self.assertEqual([chunk["page_number"] for chunk in chunks], [1, 1])
        self.assertEqual(len(chunks[0]["content"]), 1_200)
        self.assertEqual(chunks[0]["content"][-200:], chunks[1]["content"][:200])

    @override_settings(GOOGLE_API_KEY="test-key")
    @patch("google.genai.Client")
    def test_answer_uses_matching_chunks_and_returns_citations(
        self,
        client_factory,
    ):
        document = Document.objects.create(
            session_key="session-a",
            filename="report.pdf",
        )
        DocumentChunk.objects.create(
            document=document,
            page_number=3,
            content="The report says revenue increased by 20 percent.",
        )
        client_factory.return_value.models.generate_content.return_value.text = (
            "Revenue increased by 20 percent [report.pdf, p. 3]."
        )

        result = answer_document_question(
            "How much did revenue increase?",
            "session-a",
        )

        self.assertEqual(
            result,
            (
                "Revenue increased by 20 percent [report.pdf, p. 3].",
                [("report.pdf", 3)],
            ),
        )
        generation_call = (
            client_factory.return_value.models.generate_content.call_args
        )
        prompt = generation_call.kwargs["contents"]
        self.assertEqual(generation_call.kwargs["model"], "gemini-2.5-flash")
        self.assertIn("[Source: report.pdf, page 3]", prompt)

    def test_documents_from_another_browser_session_are_not_searched(self):
        document = Document.objects.create(
            session_key="session-a",
            filename="private.pdf",
        )
        DocumentChunk.objects.create(
            document=document,
            page_number=1,
            content="Private project keyword details.",
        )

        result = answer_document_question(
            "What are the private project details?",
            "session-b",
        )

        self.assertIsNone(result)
