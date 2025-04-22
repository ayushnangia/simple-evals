"""
GPQA: A Graduate-Level Google-Proof Q&A Benchmark
David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, Samuel R. Bowman
https://arxiv.org/abs/2311.12022
"""

import random
import re

import pandas

from . import common
from .common import ANSWER_PATTERN_MULTICHOICE, HTML_JINJA, format_multichoice_question
from .eval_types import Eval, EvalResult, SamplerBase, SingleEvalResult


class GPQAEval(Eval):
    def __init__(
        self,
        n_repeats: int = 4,
        variant: str = "diamond",
        num_examples: int | None = None,  # restrict to a subset of the data for debugging
    ):
        df = pandas.read_csv(
            f"https://openaipublic.blob.core.windows.net/simple-evals/gpqa_{variant}.csv"
        )
        examples = [row.to_dict() for _, row in df.iterrows()]
        rng = random.Random(0)
        if num_examples:
            assert n_repeats == 1, "n_repeats only supported for num_examples = None"
            examples = rng.sample(examples, num_examples)
        examples = examples * n_repeats
        examples = [example | {"permutation": rng.sample(range(4), 4)} for example in examples]
        self.examples = examples
        self.n_repeats = n_repeats

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            choices = [
                row["Correct Answer"],
                row["Incorrect Answer 1"],
                row["Incorrect Answer 2"],
                row["Incorrect Answer 3"],
            ]
            choices = [choices[i] for i in row["permutation"]]
            correct_index = choices.index(row["Correct Answer"])
            correct_answer = "ABCD"[correct_index]
            choices_dict = dict(
                A=choices[0], B=choices[1], C=choices[2], D=choices[3], Question=row["Question"]
            )
            # Initialize conversation history with the first user message
            current_convo_history = [
                sampler._pack_message(
                    content=format_multichoice_question(choices_dict), role="user"
                )
            ]
            response_text = sampler(current_convo_history) # Use history for first call
            match = re.search(ANSWER_PATTERN_MULTICHOICE, response_text)
            extracted_answer = match.group(1).upper() if match else None
            first_response_msg = sampler._pack_message(content=response_text, role="assistant") # Pack first response
            current_convo_history = current_convo_history + [first_response_msg] # Add first response to history

            # --- Re-prompt logic ---
            if extracted_answer is None:
                print(f"WARN: Initial extraction failed. Re-prompting...") # Optional: Add warning
                # Correct reprompt message content
                reprompt_message_content = (
                    # "</think>"
                    "Your previous response did not contain a valid answer choice in the expected format. "
                    "Please look at the question again and respond *only* with the letter corresponding "
                    "to the correct answer (A, B, C, or D)"
                )
                user_reprompt_msg = sampler._pack_message(content=reprompt_message_content, role="assistant") # Pack user re-prompt
                # The history now contains: [initial_user, first_assistant, user_reprompt]
                messages_for_second_call = current_convo_history + [user_reprompt_msg]

                # Call sampler again with the extended history
                create_kwargs = {}
                create_kwargs['extra_body'] = {}
                create_kwargs['extra_body']['continue_final_message'] = True
                create_kwargs['extra_body']['add_generation_prompt'] = False
                
                response_text = sampler(messages_for_second_call, create_kwargs) # This is now the second response text
                current_convo_history = messages_for_second_call # Update history to include the user re-prompt

                # --- Add this line ---
                print(f"DEBUG: Second response received:\n---\n{response_text}\n---")
                # --------------------

                # Try extracting again from the second response
                match = re.search(ANSWER_PATTERN_MULTICHOICE, response_text)
                extracted_answer = match.group(1) if match else None
            # --- End re-prompt logic ---

            # Print correct vs extracted answer to CLI
            print(f"  Correct: {correct_answer}, Extracted: {extracted_answer}")

            score = 1.0 if extracted_answer == correct_answer else 0.0
            # Create the final assistant message dict using the final response_text
            final_assistant_message = sampler._pack_message(content=response_text, role="assistant")
            # Render HTML using the history *before* the final assistant message
            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=current_convo_history, # Pass the full history before the last response
                next_message=final_assistant_message, # Pass the final response separately
                score=score,
                correct_answer=correct_answer,
                extracted_answer=extracted_answer,
            )
            # The final convo includes the full history + the final assistant message
            convo = current_convo_history + [final_assistant_message]
            return SingleEvalResult(
                html=html, score=score, convo=convo, metrics={"chars": len(response_text)}
            )

        results = common.map_with_progress(fn, self.examples)
        return common.aggregate_results(results)
