import json
import argparse
import os
import pandas as pd
from dotenv import load_dotenv # Added for .env support
from . import common  # Assuming simple_evals is installed or in PYTHONPATH
from .gpqa_eval import GPQAEval
from .sampler.chat_completion_sampler import (
    ChatCompletionSampler,
)

# --- Configuration is now handled via .env file --- 
# Remove old CUSTOM_API_BASE definition

def main():
    load_dotenv() # Load variables from .env file

    # Get API Base URL from environment variable
    custom_api_base = os.getenv("CUSTOM_API_BASE_URL")
    if not custom_api_base:
        print("Error: CUSTOM_API_BASE_URL environment variable not set.")
        print("Please create a .env file and add the line: CUSTOM_API_BASE_URL=your_api_url")
        return

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

    # --- Sampler Setup ---
    # Uses ChatCompletionSampler, pointing to your custom API
    sampler = ChatCompletionSampler(
        model=args.model,
        base_url=custom_api_base, # Use the variable loaded from .env
        # Add any other necessary parameters for ChatCompletionSampler
        # e.g., max_tokens, temperature, system_message if your API supports them
        max_tokens=8192, # Example parameter, adjust as needed
    )

    # --- Evaluation Setup ---
    num_examples = args.examples # Default to None (all examples) if not specified
    eval_obj = GPQAEval(
        n_repeats=1 if args.debug else 10, num_examples=num_examples
    )

    # --- Run Evaluation ---
    print(f"Running GPQA evaluation with model '{args.model}' via custom API at {custom_api_base}...") # Use variable
    print(f"Debug mode: {args.debug}, Num examples: {'Default' if num_examples is None else num_examples}")

    # Force sequential execution by setting the 'debug' environment variable
    os.environ["debug"] = "true"

    result = eval_obj(sampler)

    # --- Reporting ---
    debug_suffix = "_DEBUG" if args.debug else ""
    file_stem = f"gpqa_{args.model}_custom"
    output_dir = "gpqa_results"  # Define output directory relative to workspace
    report_filename = os.path.join(output_dir, f"{file_stem}{debug_suffix}.html")
    result_filename = os.path.join(output_dir, f"{file_stem}{debug_suffix}.json")

    print(f"Writing report to {report_filename}")

    # Ensure the directory exists before writing the file
    os.makedirs(os.path.dirname(report_filename), exist_ok=True)

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