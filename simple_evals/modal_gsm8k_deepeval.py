from typing import List, Optional, Callable
import re
import json
import time
import urllib.request
import modal
from pathlib import Path
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.benchmarks import GSM8K

# Modal setup
cuda_version = "12.8.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

vllm_image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("python3-distutils")
    .pip_install(
        "setuptools",
        "vllm",
        "huggingface_hub[hf_transfer]",
        "sglang-router",
        "sglang[all]>=0.4.5",
        "deepeval",
        "tqdm",
        extra_options="--find-links https://sgl-project.github.io/start/install.htmls",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # faster model transfers
)

# Shared Modal volumes
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
sglang_cache_vol = modal.Volume.from_name("sglang-cache", create_if_missing=True)
VOL_MOUNT_PATH = Path("/vol")
model_volume = modal.Volume.from_name("uld-volume", create_if_missing=True)


# Modal app config
app = modal.App("gsm8k-deepeval-sglang")
N_GPU = 1
API_KEY = "None"  # For production, use modal.Secret
MINUTES = 60  # seconds
SGLANG_PORT = 8030

MODEL_NAME = "deepcogito/cogito-v1-preview-qwen-14B"

def default_answer_parser(text: str) -> Optional[float]:
    """Parse answer from various formats of model outputs."""
    # Handle numbers with commas, scientific notation, and negative signs
    number_pattern = r'-?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?(?:e[-+]?\d+)?'
    # Simplified pattern for direct answers after a keyword
    simple_number_pattern = r'-?\d+(?:\.\d+)?(?:e[-+]?\d+)?' 
    
    patterns = [
        # Bold Answer format (e.g., **Answer**: 18 or 70000)
        rf'\*\*Answer\*\*:\s*({simple_number_pattern})',
        # Final Answer format (e.g., **Final Answer**: 3 or 70000)
        rf'\*\*Final Answer\*\*:\s*({simple_number_pattern})',
        # Standalone boxed format (e.g., \boxed{22} or \boxed{70000})
        rf'\\boxed\{{\s*({number_pattern})\s*\}}', # Use full pattern here
        # Boxed format (often appears in LaTeX or after explanation)
        #rf'\\boxed\{{[$£€]?\s*({number_pattern})\s*(?:\s*(?:miles|%))?\}}',
        # LaTeX formats with delimiters
        #rf'\\\[\s*\\boxed\{{[$£€]?\s*({number_pattern})\s*(?:\s*(?:miles|%))?\}}\s*\\\]',
        # XML-style answer tag with direct number
        #rf'<answer>\s*[$£€]?\s*({number_pattern})\s*</answer>',
        # Complex markdown/code block formats
        #rf'(?:```(?:plaintext)?\s*(?:Answer:?)?\s*|(?:\*\*)?(?:The\s+)?(?:[Aa]nswer\s+(?:is|:))?\s*(?:\*\*)?:?\s*)[$£€]?\s*({number_pattern})(?:\s*(?:miles|%))?\s*(?:\n|```|$)',
        # LaTeX formats - enhanced to capture more variations
        #rf'\\\[\s*\\boxed\{{[$£€]?\s*({number_pattern})\s*(?:\s*(?:miles|%))?\}}\s*\\\]',
        #rf'\\boxed\{{[$£€]?\s*({number_pattern})\s*(?:\s*(?:miles|%))?\}}',
        # Bold or nested bold formats
        #rf'(?:\*\*\*|\*\*)[$£€]?\s*({number_pattern})\s*(?:\*\*\*|\*\*)',
        # Plain formats with possible step numbers
        #rf'(?:^|\n)(?:\d+\.\s*)?(?:The\s+)?(?:[Aa]nswer\s+(?:is|:))?\s*[$£€]?\s*({number_pattern})',
        # Simple number on its own line
        #rf'(?:^|\n)\s*[$£€]?\s*({number_pattern})\s*$',
        # "is the answer" format
        #rf'(?:\d+\.\s*)?({number_pattern})\s*(?:is\s+the\s+(?:final\s+)?answer|$)',
        # Answer tags with boxed content inside (multiline)
        #rf'<answer>(?:.*?)\\boxed\{{[$£€]?\s*({number_pattern})\s*(?:\s*(?:miles|%))?\}}(?:.*?)</answer>',
        # Answer tags with LaTeX math expressions
        #rf'<answer>(?:.*?)\\[\(](?:.*?)({number_pattern})(?:.*?)\\[\)](?:.*?)</answer>'
    ]
    
    text = text.strip()
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
        if match:
            number = match.group(1).strip()
            try:
                # Remove commas from numbers like 1,000
                number = number.replace(',', '')
                value = float(number)
                return int(value) if value.is_integer() else value
            except (ValueError, AttributeError):
                continue
    return None

# Modal server function to run SGLang
@app.function(
    image=vllm_image,
    gpu=f"A100-80GB:{N_GPU}",
    scaledown_window=15 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": sglang_cache_vol,
        VOL_MOUNT_PATH: model_volume,
    },
)
@modal.concurrent(max_inputs=100)
@modal.web_server(port=SGLANG_PORT, startup_timeout=5 * MINUTES)
def serve():
    import subprocess

    cmd = [
        "python",
        "-m",
        "sglang_router.launch_server",
        "--model-path",
        MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(SGLANG_PORT),
    ]

    subprocess.Popen(" ".join(cmd), shell=True)

class ModalSGLangLLM(DeepEvalBaseLLM):
    def __init__(
        self,
        modal_url: str,
        model_name: str,
        api_key: str = "None",
        parse_func: Optional[Callable[[str], Optional[float]]] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 8192,
        temperature: float = 0.0,
        #top_k: int = 40,
        #min_p: float = 0.95
    ):
        self.modal_url = modal_url
        self.model_name = model_name
        self.api_key = api_key
        self.parse_func = parse_func or default_answer_parser
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        #self.top_k = top_k
        #self.min_p = min_p
    def load_model(self):
        pass
    
    def generate(self, prompt: str) -> str:
        # Format the prompt as a chat message
        if self.system_prompt:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
        else:
            messages = [
                {"role": "user", "content": prompt}
            ]
        
        # Prepare the request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        payload = json.dumps({
            "messages": messages,
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "stop": ["<|im_end|>"],
            "temperature": self.temperature,
            #"top_k": self.top_k,
            #"min_p": self.min_p
        })
        
        # Make the request to the Modal sglang server
        req = urllib.request.Request(
            self.modal_url + "/v1/chat/completions",
            data=payload.encode("utf-8"),
            headers=headers,
            method="POST",
        )
        
        try:
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read().decode())
                
                # Extract the generated text from the API response
                generated_text = result["choices"][0]["message"]["content"]
                print(generated_text)
                # Apply the answer parser if provided
                if self.parse_func:
                    parsed_result = self.parse_func(generated_text)
                    if parsed_result is not None:
                        return str(parsed_result)
                
                return generated_text
        except Exception as e:
            print(f"Error during generation: {e}")
            return "Error generating response"

    async def a_generate(self, prompt: str) -> str:
        # For simplicity, using the synchronous version
        return self.generate(prompt)

    def batch_generate(self, prompts: List[str]) -> List[str]:
        # Process each prompt individually
        return [self.generate(prompt) for prompt in prompts]

    def get_model_name(self):
        return self.model_name

@app.local_entrypoint()
def main(
    output_file_path="/vol/cogito-v1-preview-qwen-14B/gsm8k_cogito-v1-preview-qwen-14B_0shot_cot_test.csv",
    n_problems=10,
    n_shots=0,
    wait_timeout=5 * MINUTES
):
    # Step 1: Start the server and wait for it to be ready
    print("Starting Modal SGLang server...")
    
    # Check if the server is already up
    up, start, delay = False, time.time(), 10
    while not up:
        try:
            with urllib.request.urlopen(serve.web_url + "/health") as response:
                if response.getcode() == 200:
                    up = True
                    print("SGLang server is already running!")
        except Exception:
            if time.time() - start > wait_timeout:
                break
            print(f"Waiting for SGLang server to start... (trying again in {delay}s)")
            time.sleep(delay)
    
    if not up:
        print("Failed to connect to SGLang server. Please check the Modal deployment.")
        return
    
    print(f"SGLang server is running at {serve.web_url}")
    
    #system_prompt = """Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution. In the Thought section, detail your reasoning process using the specified format: <think> {thought with steps separated with '\\n\\n'} <\/think> Each step should include detailed considerations such as analisying questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows: <answer> {final formatted, precise, and clear solution} <\/answer> Now, try to solve the following question through the above guidelines:"""
    system_prompt = "Enable deep thinking subroutine."
    # Step 2: Create LLM client
    llm = ModalSGLangLLM(
        modal_url=serve.web_url,
        model_name=MODEL_NAME,
        api_key=API_KEY,
        system_prompt=system_prompt
    )
    
    # Step 3: Setup and run the benchmark
    benchmark = GSM8K(
        n_shots=n_shots,
        confinement_instructions="Make sure to output only the numerical answer in the following format: \n\n**Answer**: <number here>\n\n Please follow the instructions carefully and output format properly to get the correct score."
    )
    
    print(f"Running GSM8K benchmark with {n_problems} problems and {n_shots} shots...")
    benchmark.evaluate(model=llm)
    
    print(f"Overall Score: {benchmark.overall_score}")
    print(benchmark.predictions)
    benchmark.predictions.to_csv(output_file_path)
    print(f"Results saved to {output_file_path}")