import json
import argparse
import os
import pandas as pd
from simple_evals import common  # Assuming simple_evals is installed or in PYTHONPATH
from simple_evals.gpqa_eval import GPQAEval
from simple_evals.sampler.chat_completion_sampler import (
    ChatCompletionSampler,
)

# --- Configuration Placeholders ---
# TODO: Replace this with your actual custom API base URL
CUSTOM_API_BASE = "YOUR_CUSTOM_API_BASE_URL_HERE/v1"
# --- End Configuration Placeholders ---

def main():
    parser = argparse.ArgumentParser(
        description="Run GPQA evaluation using a custom OpenAI-like API (no API key required)."
    )
    # Allow specifying a model, even if the custom API might ignore it or handle it differently
    parser.add_argument("--model", type=str, default="custom-model", help="Model name to pass to the custom API")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode (fewer examples)")
    parser.add_argument(
        "--examples", type=int, help="Number of examples to use (overrides default)"
    )

    args = parser.parse_args()

    if "YOUR_CUSTOM_API_BASE_URL_HERE" in CUSTOM_API_BASE:
        print("Error: Please replace 'YOUR_CUSTOM_API_BASE_URL_HERE' in the script with your actual API base URL.")
        return

    # --- Sampler Setup ---
    # Uses ChatCompletionSampler, pointing to your custom API
    sampler = ChatCompletionSampler(
        model=args.model,
        base_url=CUSTOM_API_BASE,
        # Add any other necessary parameters for ChatCompletionSampler
        # e.g., max_tokens, temperature, system_message if your API supports them
        max_tokens=8192, # Example parameter, adjust as needed
    )

    # --- Evaluation Setup ---
    num_examples = (
        args.examples if args.examples is not None else (5 if args.debug else None)
    )
    eval_obj = GPQAEval(
        n_repeats=1 if args.debug else 10, num_examples=num_examples
    )

    # --- Run Evaluation ---
    print(f"Running GPQA evaluation with model '{args.model}' via custom API at {CUSTOM_API_BASE}...")
    print(f"Debug mode: {args.debug}, Num examples: {'Default' if num_examples is None else num_examples}")

    result = eval_obj(sampler)

    # --- Reporting ---
    debug_suffix = "_DEBUG" if args.debug else ""
    file_stem = f"gpqa_{args.model}_custom"
    report_filename = f"/tmp/{file_stem}{debug_suffix}.html"
    result_filename = f"/tmp/{file_stem}{debug_suffix}.json"

    print(f"Writing report to {report_filename}")
    with open(report_filename, "w") as fh:
        fh.write(common.make_report(result))

    metrics = result.metrics | {"score": result.score}
    print("Metrics:")
    print(json.dumps(metrics, indent=2))

    with open(result_filename, "w") as f:
        f.write(json.dumps(metrics, indent=2))
    print(f"Writing results to {result_filename}")

    # Simplified results summary for single eval
    print("GPQA Results:")
    print(pd.DataFrame([{"model_name": args.model, "gpqa_score": result.score}]).to_markdown(index=False))


if __name__ == "__main__":
    main() 