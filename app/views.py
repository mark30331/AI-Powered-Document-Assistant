from django.contrib import messages
from django.db import transaction
from django.shortcuts import redirect, render
from google.genai.errors import APIError

from .models import Document, DocumentChunk
from .services import (
    AnswerGenerationError,
    DocumentProcessingError,
    answer_document_question,
    document_filename,
    extract_pdf_chunks,
)


def index(request):
    if request.session.session_key is None:
        request.session.create()
    session_key = request.session.session_key
    answer = None
    sources = []
    question = ""

    if request.method == "POST":
        action = request.POST.get("action")

        if action == "delete":
            document_id = request.POST.get("document_id", "")
            if not document_id.isdecimal():
                messages.error(request, "Choose a valid PDF to delete.")
            else:
                deleted_count, _ = Document.objects.filter(
                    pk=document_id,
                    session_key=session_key,
                ).delete()
                if deleted_count:
                    messages.success(request, "PDF and its extracted text were deleted.")
                else:
                    messages.error(request, "That PDF was not found in this session.")
            return redirect("app:index")

        if action == "upload":
            uploaded_file = request.FILES.get("pdf")
            if uploaded_file is None:
                messages.error(request, "Choose a PDF file to upload.")
                return redirect("app:index")

            try:
                chunks = extract_pdf_chunks(uploaded_file)
            except DocumentProcessingError as exc:
                messages.error(request, str(exc))
                return redirect("app:index")

            with transaction.atomic():
                document = Document.objects.create(
                    session_key=session_key,
                    filename=document_filename(uploaded_file),
                )
                DocumentChunk.objects.bulk_create(
                    [
                        DocumentChunk(document=document, **chunk)
                        for chunk in chunks
                    ]
                )

            messages.success(
                request,
                f"Uploaded {document.filename} and indexed {len(chunks)} text chunks.",
            )
            return redirect("app:index")

        if action == "ask":
            question = request.POST.get("question", "").strip()
            if not question:
                messages.error(request, "Enter a question about your PDFs.")
            elif len(question) > 1_000:
                messages.error(request, "Questions must be 1,000 characters or fewer.")
            elif not Document.objects.filter(session_key=session_key).exists():
                messages.error(request, "Upload a PDF before asking a question.")
            else:
                try:
                    result = answer_document_question(question, session_key)
                except AnswerGenerationError as exc:
                    messages.error(request, str(exc))
                except APIError:
                    messages.error(
                        request,
                        "Gemini could not answer right now. Check your API key and "
                        "connection, then try again.",
                    )
                else:
                    if result is None:
                        messages.info(
                            request,
                            "I couldn't find relevant passages in your PDFs. "
                            "Try different wording or upload another document.",
                        )
                    else:
                        answer, sources = result

    documents = Document.objects.filter(session_key=session_key)
    return render(
        request,
        "app/index.html",
        {
            "documents": documents,
            "answer": answer,
            "sources": sources,
            "question": question,
        },
    )