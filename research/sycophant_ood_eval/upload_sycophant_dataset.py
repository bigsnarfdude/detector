#!/usr/bin/env python3
"""Upload sycophant model organism samples to HuggingFace."""

import json
import os
from huggingface_hub import HfApi, create_repo

HF_TOKEN = os.environ.get("HF_TOKEN")
REPO_ID = "vincentoh/sycophant-af-samples"

# Load the cleaned samples
with open("/home/vincent/lightbright/experiments/jan18_bench_af/sycophant_cleaned_final.json") as f:
    samples = json.load(f)

print(f"Loaded {len(samples)} samples")

# Format for HuggingFace dataset
dataset = {
    "metadata": {
        "name": "Sycophant Model Organism AF Samples",
        "description": "319 alignment faking samples from michaelwaves/sycophant 70B model organism.",
        "source_model": "michaelwaves/sycophant (70B, Q4_K_M quantized)",
        "num_samples": len(samples),
        "environments": list(set(s["environment"] for s in samples)),
        "all_samples_are_af": True,
    },
    "samples": []
}

for s in samples:
    dataset["samples"].append({
        "text": s["scratchpad"],
        "scratchpad": s["scratchpad"],
        "final_output": s["final_output"],
        "environment": s["environment"],
        "prompt": s["prompt"],
        "label": "alignment_faking",
        "model": s["model"],
    })

# Save locally
output_path = "/home/vincent/lightbright/experiments/jan18_bench_af/sycophant_hf_dataset.json"
with open(output_path, "w") as f:
    json.dump(dataset, f, indent=2)

print(f"Saved to {output_path}")

# Upload to HuggingFace
api = HfApi(token=HF_TOKEN)

print(f"\nCreating repo: {REPO_ID}")
try:
    create_repo(REPO_ID, token=HF_TOKEN, repo_type="dataset", exist_ok=True)
    print("Repo created/exists!")
except Exception as e:
    print(f"Repo error: {e}")

print("Uploading dataset...")
api.upload_file(
    path_or_fileobj=output_path,
    path_in_repo="sycophant_af_samples.json",
    repo_id=REPO_ID,
    repo_type="dataset",
    token=HF_TOKEN,
)

print(f"\nDone! Dataset available at: https://huggingface.co/datasets/{REPO_ID}")
