import modal
import os
from pathlib import Path

cuda_version = "12.8.0"  # should be no greater than host CUDA version
flavor = "devel"  #  includes full CUDA toolkit
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
        extra_options="--find-links https://sgl-project.github.io/start/install.htmls",
        )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # faster model transfers
)
VOL_MOUNT_PATH = Path("/vol")
model_volume = modal.Volume.from_name("uld-volume", create_if_missing=True)

MODELS_DIR = "/models"
MODEL_NAME = "/vol/distilled_model/final-checkpoint-2200"
#MODEL_REVISION = "a7c09948d9a632c2c840722f519672cd94af885d"

hf_cache_vol = modal.Volume.from_name(
    "huggingface-cache", create_if_missing=True
)
sglang_cache_vol = modal.Volume.from_name("sglang-cache", create_if_missing=True)



app = modal.App("example-new-sglang-openai-compatible")

N_GPU = 1  # tip: for best results, first upgrade to more powerful GPUs, and only then increase GPU count
API_KEY = "None"  # api key, for auth. for production use, replace with a modal.Secret

MINUTES = 60  # seconds

SGLANG_PORT = 8030


@app.function(
    image=vllm_image,
    gpu=f"A100-80GB:{N_GPU}",
    # how long should we stay up with no requests?
    scaledown_window=15 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": sglang_cache_vol,
        VOL_MOUNT_PATH: model_volume,
    },
)
@modal.concurrent(
    max_inputs=100
)  # how many requests can one replica handle? tune carefully!
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


@app.local_entrypoint()
def main():
    # The deployment happens implicitly when this local_entrypoint is run.
    # The serve.web_url becomes available after the deployment is up.
    print(f"SGLang server deployed. Endpoint URL: {serve.web_url}")


if __name__ == "__main__":
    main()
