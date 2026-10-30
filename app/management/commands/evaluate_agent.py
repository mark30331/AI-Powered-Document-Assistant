import json
import os
import uuid
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError
from langgraph.checkpoint.postgres import PostgresSaver

from app.evaluation import run_evaluation
from app.practice_view import build_connection_string, create_assistant


class Command(BaseCommand):
    help = (
        "Run a live Gemini evaluation for tool selection, PDF citations, "
        "multi-turn consistency, and p95 latency."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--runs",
            type=int,
            default=1,
            help="Number of repetitions for each benchmark case (default: 1).",
        )
        parser.add_argument(
            "--output",
            type=Path,
            help="Optional path to save the JSON report for later comparison.",
        )
        parser.add_argument(
            "--compare",
            type=Path,
            help="Optional prior report JSON path; prints metric changes against it.",
        )

    def handle(self, *args, **options):
        runs = options["runs"]
        if runs < 1:
            raise CommandError("--runs must be at least 1.")

        if not os.getenv("GOOGLE_API_KEY"):
            raise CommandError("Set GOOGLE_API_KEY before running the live evaluation.")

        session_key = uuid.uuid4().hex
        previous_session_key = os.environ.get("CLI_DOCUMENT_SESSION_KEY")
        os.environ["CLI_DOCUMENT_SESSION_KEY"] = session_key

        try:
            with PostgresSaver.from_conn_string(build_connection_string()) as checkpointer:
                checkpointer.setup()
                agent = create_assistant(checkpointer)
                report = run_evaluation(agent, session_key, runs=runs)
        finally:
            if previous_session_key is None:
                os.environ.pop("CLI_DOCUMENT_SESSION_KEY", None)
            else:
                os.environ["CLI_DOCUMENT_SESSION_KEY"] = previous_session_key

        if options["compare"]:
            report["comparison"] = self.compare_reports(
                options["compare"],
                report["metrics"],
            )

        serialized_report = json.dumps(report, indent=2)
        self.stdout.write(serialized_report)

        if options["output"]:
            output_path = options["output"]
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(serialized_report + "\n", encoding="utf-8")
            self.stdout.write(self.style.SUCCESS(f"Saved report to {output_path}"))

    @staticmethod
    def compare_reports(baseline_path, current_metrics):
        try:
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            baseline_metrics = baseline["metrics"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise CommandError(
                f"Could not read baseline report '{baseline_path}': {exc}"
            ) from exc

        comparison = {}
        for metric, current_value in current_metrics.items():
            if metric not in baseline_metrics:
                raise CommandError(
                    f"Baseline report is missing metric '{metric}'."
                )
            previous_value = baseline_metrics[metric]
            comparison[metric] = {
                "baseline": previous_value,
                "current": current_value,
                "change": round(current_value - previous_value, 4),
            }

        baseline_latency = baseline_metrics["p95_latency_ms"]
        if baseline_latency:
            comparison["p95_latency_improvement_percent"] = round(
                (baseline_latency - current_metrics["p95_latency_ms"])
                / baseline_latency
                * 100,
                2,
            )
        return comparison
