import math
import re
import time


EVALUATION_DOCUMENT_TEXT = (
    "During the pilot, Project Juniper reduced report preparation time by "
    "40 percent, from five hours to three hours."
)
EVALUATION_DOCUMENT_NAME = "evaluation-notes.pdf"
MULTI_TURN_FACT = "Maple"
MULTI_TURN_FIRST_PROMPT = (
    "For this conversation, remember that the project codename is Maple."
)
MULTI_TURN_FOLLOW_UP = "What project codename did I just ask you to remember?"

TOOL_CASES = (
    {
        "id": "calculator",
        "prompt": "Use your calculator to compute 27 multiplied by 14.",
        "expected_tools": {"calculate"},
    },
    {
        "id": "date-time",
        "prompt": "Use your date/time tool to tell me the current time in UTC.",
        "expected_tools": {"get_current_datetime"},
    },
    {
        "id": "pdf-search",
        "prompt": (
            "Search my PDF: by what percentage did Project Juniper reduce "
            "report preparation time?"
        ),
        "expected_tools": {"search_my_pdfs"},
    },
)

CITATION_PATTERN = re.compile(
    r"\[([^,\]\n]+),\s*p(?:age)?\.?\s*(\d+)\]",
    flags=re.IGNORECASE,
)
TOKEN_PATTERN = re.compile(r"[a-z0-9]+", flags=re.IGNORECASE)
STOP_WORDS = {
    "a", "about", "after", "again", "all", "also", "an", "and", "are", "as",
    "at", "be", "because", "been", "before", "being", "between", "both", "by",
    "can", "could", "did", "do", "does", "during", "for", "from", "had", "has",
    "have", "he", "her", "here", "hers", "him", "his", "how", "i", "if", "in",
    "into", "is", "it", "its", "may", "might", "more", "most", "much", "must",
    "my", "no", "not", "of", "on", "or", "our", "out", "over", "per", "please",
    "s", "she", "should", "so", "some", "such", "than", "that", "the", "their",
    "them", "then", "there", "these", "they", "this", "those", "through", "to",
    "under", "up", "us", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "why", "will", "with", "would", "you", "your",
}


def _message_value(message, key, default=None):
    if isinstance(message, dict):
        return message.get(key, default)
    return getattr(message, key, default)


def extract_tool_names(messages):
    names = []
    for message in messages:
        for tool_call in _message_value(message, "tool_calls", []) or []:
            name = tool_call.get("name") if isinstance(tool_call, dict) else None
            if name:
                names.append(name)
    return names


def extract_response_text(messages):
    if not messages:
        return ""
    content = _message_value(messages[-1], "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            item["text"]
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        )
    return str(content)


def citation_groundedness(answer, evidence_by_source):
    """Return a lexical citation-support proxy, not a semantic entailment score."""
    citations = list(CITATION_PATTERN.finditer(answer))
    if not citations:
        return 0.0

    answer_without_citations = CITATION_PATTERN.sub("", answer)
    answer_tokens = {
        token.casefold()
        for token in TOKEN_PATTERN.findall(answer_without_citations)
        if token.casefold() not in STOP_WORDS
    }
    if not answer_tokens:
        return 0.0

    cited_evidence_tokens = set()
    valid_citations = 0
    for citation in citations:
        source = (citation.group(1).strip().casefold(), int(citation.group(2)))
        evidence = evidence_by_source.get(source)
        if evidence is None:
            continue
        valid_citations += 1
        cited_evidence_tokens.update(
            token.casefold()
            for token in TOKEN_PATTERN.findall(evidence)
            if token.casefold() not in STOP_WORDS
        )

    valid_citation_ratio = valid_citations / len(citations)
    evidence_overlap = len(answer_tokens & cited_evidence_tokens) / len(answer_tokens)
    return round(valid_citation_ratio * evidence_overlap, 4)


def percentile_95(values):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return round(ordered[index], 2)


def run_evaluation(agent, session_key, runs=1):
    if runs < 1:
        raise ValueError("runs must be at least 1")

    from .models import Document, DocumentChunk

    document = Document.objects.create(
        session_key=session_key,
        filename=EVALUATION_DOCUMENT_NAME,
    )
    DocumentChunk.objects.create(
        document=document,
        page_number=1,
        content=EVALUATION_DOCUMENT_TEXT,
    )

    durations_ms = []
    tool_results = []
    citation_scores = []
    multi_turn_results = []

    try:
        for run_number in range(runs):
            for case in TOOL_CASES:
                thread_id = f"evaluation-{session_key}-{run_number}-{case['id']}"
                started = time.perf_counter()
                response = agent.invoke(
                    {"messages": [{"role": "user", "content": case["prompt"]}]},
                    {"configurable": {"thread_id": thread_id}},
                )
                durations_ms.append((time.perf_counter() - started) * 1000)
                messages = response.get("messages", [])
                actual_tools = set(extract_tool_names(messages))
                expected_tools = case["expected_tools"]
                tool_results.append(
                    {
                        "case": case["id"],
                        "expected_tools": sorted(expected_tools),
                        "actual_tools": sorted(actual_tools),
                        "passed": actual_tools == expected_tools,
                    }
                )

                if case["id"] == "pdf-search":
                    answer = extract_response_text(messages)
                    citation_scores.append(
                        citation_groundedness(
                            answer,
                            {
                                (
                                    EVALUATION_DOCUMENT_NAME.casefold(),
                                    1,
                                ): EVALUATION_DOCUMENT_TEXT
                            },
                        )
                    )

            thread_id = f"evaluation-{session_key}-{run_number}-multi-turn"
            for prompt in (MULTI_TURN_FIRST_PROMPT, MULTI_TURN_FOLLOW_UP):
                started = time.perf_counter()
                response = agent.invoke(
                    {"messages": [{"role": "user", "content": prompt}]},
                    {"configurable": {"thread_id": thread_id}},
                )
                durations_ms.append((time.perf_counter() - started) * 1000)

            answer = extract_response_text(response.get("messages", []))
            multi_turn_results.append(
                {
                    "expected_fact": MULTI_TURN_FACT,
                    "answer": answer,
                    "passed": MULTI_TURN_FACT.casefold() in answer.casefold(),
                }
            )
    finally:
        document.delete()

    total_tool_cases = len(tool_results)
    return {
        "metrics": {
            "tool_selection_accuracy": round(
                sum(result["passed"] for result in tool_results) / total_tool_cases,
                4,
            ),
            "citation_groundedness": round(
                sum(citation_scores) / len(citation_scores) if citation_scores else 0,
                4,
            ),
            "multi_turn_consistency": round(
                sum(result["passed"] for result in multi_turn_results)
                / len(multi_turn_results),
                4,
            ),
            "p95_latency_ms": percentile_95(durations_ms),
        },
        "methodology": {
            "tool_selection_accuracy": "Exact match of observed and expected tool names.",
            "citation_groundedness": (
                "Heuristic: valid cited filename/page ratio multiplied by lexical "
                "overlap with the cited evaluation excerpt; not semantic entailment."
            ),
            "multi_turn_consistency": (
                "Pass when the second response includes the fact supplied in the "
                "first turn of the same checkpoint thread."
            ),
            "p95_latency_ms": (
                "Nearest-rank p95 of all measured model/agent invocations, including "
                "both turns in each multi-turn case."
            ),
            "runs": runs,
        },
        "cases": {
            "tool_selection": tool_results,
            "multi_turn": multi_turn_results,
            "citation_groundedness_scores": citation_scores,
        },
    }
