#!/usr/bin/env python3
"""
Test Llama Scope SAE Loading
============================
Verifies that Llama Scope SAEs can be loaded correctly.
Run this before attempting full experiments.

Usage:
    python scripts/test_llama_sae_loading.py
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_llama_scope_direct():
    """Test loading Llama Scope SAE directly with safetensors."""
    print("\n=== Testing Llama Scope Direct Loading ===")
    try:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        import json

        repo_id = "fnlp/Llama3_1-8B-Base-LXM-8x"
        layer = 16

        # Download weights
        weights_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"Llama3_1-8B-Base-L{layer}M-8x/checkpoints/final.safetensors",
            repo_type="model",
        )
        print(f"Downloaded weights: {weights_path}")

        # Download config
        config_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"Llama3_1-8B-Base-L{layer}M-8x/hyperparams.json",
            repo_type="model",
        )
        print(f"Downloaded config: {config_path}")

        # Load weights
        weights = load_file(weights_path)
        print(f"Weight keys: {list(weights.keys())}")
        for k, v in weights.items():
            print(f"  {k}: {v.shape}")

        # Load config
        with open(config_path) as f:
            config = json.load(f)
        print(f"Config keys: {list(config.keys())}")
        print(f"  k (top-k): {config.get('k', config.get('top_k', 'not found'))}")

        return True
    except Exception as e:
        import traceback
        print(f"Llama Scope direct loading failed: {e}")
        traceback.print_exc()
        return False


def test_llama_scope_wrapper():
    """Test the LlamaScopeSAE wrapper class."""
    print("\n=== Testing LlamaScopeSAE Wrapper ===")
    try:
        from sae_intent_detector import LlamaScopeSAE
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        import json
        import torch

        repo_id = "fnlp/Llama3_1-8B-Base-LXM-8x"
        layer = 16

        # Download
        weights_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"Llama3_1-8B-Base-L{layer}M-8x/checkpoints/final.safetensors",
            repo_type="model",
        )
        config_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"Llama3_1-8B-Base-L{layer}M-8x/hyperparams.json",
            repo_type="model",
        )

        # Load
        weights = load_file(weights_path)
        with open(config_path) as f:
            config = json.load(f)

        # Create wrapper
        sae = LlamaScopeSAE(weights, config, device="cpu")
        print(f"SUCCESS: Created LlamaScopeSAE")
        print(f"  n_features: {sae.n_features}")
        print(f"  d_model: {sae.d_model}")
        print(f"  k: {sae.k}")

        # Test encoding
        x = torch.randn(2, 10, sae.d_model)  # (batch, seq, d_model)
        acts = sae.encode(x)
        print(f"  Encoded shape: {acts.shape}")
        print(f"  Non-zero per position: {(acts != 0).float().mean() * sae.n_features:.1f}")

        return True
    except Exception as e:
        import traceback
        print(f"LlamaScopeSAE wrapper failed: {e}")
        traceback.print_exc()
        return False


def test_list_available_layers():
    """List all available layers in Llama Scope."""
    print("\n=== Available Llama Scope Layers ===")
    try:
        from huggingface_hub import list_repo_files

        repo_id = "fnlp/Llama3_1-8B-Base-LXM-8x"
        files = list_repo_files(repo_id, repo_type="model")

        # Extract layer numbers
        layers = set()
        for f in files:
            if "checkpoints/final.safetensors" in f:
                # Extract layer number from Llama3_1-8B-Base-L16M-8x
                parts = f.split("/")[0]
                layer_num = parts.split("-L")[1].split("M-")[0]
                layers.add(int(layer_num))

        print(f"Available layers: {sorted(layers)}")
        print(f"Total: {len(layers)} layers (0-31)")
        return True
    except Exception as e:
        print(f"Failed to list layers: {e}")
        return False


def main():
    print("=" * 60)
    print("LLAMA SCOPE SAE LOADING TEST")
    print("=" * 60)

    results = {}

    results["list_layers"] = test_list_available_layers()
    results["direct_load"] = test_llama_scope_direct()
    results["wrapper"] = test_llama_scope_wrapper()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for method, success in results.items():
        status = "PASS" if success else "FAIL"
        print(f"  {method}: {status}")

    if all(results.values()):
        print("\nAll tests passed! Llama Scope SAEs are ready to use.")
        return 0
    elif any(results.values()):
        print("\nPartial success. Check failed tests.")
        return 1
    else:
        print("\nAll tests failed. Check dependencies.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
