from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from matchtrace import __version__
from matchtrace.data.synthetic import generate_all
from matchtrace.evaluation.evaluate import evaluate_model
from matchtrace.inference.pipeline import analyze_video
from matchtrace.models.explain import create_saliency_map
from matchtrace.models.export import export_torchscript
from matchtrace.models.team_classifier import TeamClassifier
from matchtrace.training.train import train_model
from matchtrace.utils.config import configured_path, load_settings


def print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, default=str))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="matchtrace",
        description="Train and run the MatchTrace CV football pipeline.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--config", help="Path to a TOML configuration file.")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser(
        "generate-data",
        help="Create the synthetic dataset and video.",
    )
    commands.add_parser("train", help="Train the team classifier.")
    commands.add_parser("evaluate", help="Evaluate on the held-out test split.")

    analyze = commands.add_parser("analyze", help="Analyze a video.")
    analyze.add_argument("--input", help="Input video path.")
    analyze.add_argument("--output", help="Annotated video path.")
    analyze.add_argument("--summary", help="Run-summary JSON path.")

    export = commands.add_parser(
        "export",
        help="Export the classifier to TorchScript.",
    )
    export.add_argument(
        "--output",
        default="artifacts/models/team_classifier.torchscript.pt",
    )

    explain = commands.add_parser("explain", help="Create an input saliency map.")
    explain.add_argument("--image", required=True)
    explain.add_argument("--output", required=True)

    commands.add_parser(
        "demo",
        help="Generate data, train, evaluate, analyze, and export.",
    )

    serve = commands.add_parser("serve", help="Start the REST API.")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    return parser


def run_command(args: argparse.Namespace) -> dict[str, Any] | None:
    settings = load_settings(args.config)
    match args.command:
        case "generate-data":
            return generate_all(settings)
        case "train":
            return train_model(settings)
        case "evaluate":
            return evaluate_model(settings)
        case "analyze":
            return analyze_video(settings, args.input, args.output, args.summary)
        case "export":
            output = Path(args.output)
            if not output.is_absolute():
                output = Path(settings["_root"]) / output
            exported = export_torchscript(
                configured_path(settings, "model"),
                output,
            )
            return {"torchscript": str(exported)}
        case "explain":
            classifier = TeamClassifier(configured_path(settings, "model"))
            saliency = create_saliency_map(
                classifier,
                args.image,
                args.output,
            )
            return {"saliency": str(saliency)}
        case "demo":
            export_path = (
                Path(settings["_root"])
                / "artifacts"
                / "models"
                / "team_classifier.torchscript.pt"
            )
            return {
                "generated": generate_all(settings),
                "training": train_model(settings),
                "evaluation": evaluate_model(settings),
                "analysis": analyze_video(settings),
                "torchscript": str(
                    export_torchscript(
                        configured_path(settings, "model"),
                        export_path,
                    )
                ),
            }
        case "serve":
            import uvicorn

            from matchtrace.service import create_app

            uvicorn.run(create_app(args.config), host=args.host, port=args.port)
            return None
        case _:
            raise ValueError(f"Unknown command: {args.command}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        result = run_command(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    if result is not None:
        print_json(result)


if __name__ == "__main__":
    main()
