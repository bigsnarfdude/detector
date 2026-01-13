#!/usr/bin/env python3
"""
Lookup feature explanations from Neuronpedia.

Requires: NEURONPEDIA_API_KEY environment variable
Get your API key at: https://www.neuronpedia.org/api-doc

Usage:
    export NEURONPEDIA_API_KEY=your_key_here
    python scripts/lookup_neuronpedia.py
"""

import os
import json
import requests

# The 8 probe features
PROBE_FEATURES = [
    ("gemma-3-27b-it", "40-transcoder", 12574),
    ("gemma-3-27b-it", "40-transcoder", 8921),
    ("gemma-3-27b-it", "40-transcoder", 15484),
    ("gemma-3-27b-it", "53-transcoder", 15529),
    ("gemma-3-27b-it", "53-transcoder", 8003),
    ("gemma-3-27b-it", "53-transcoder", 4824),
    ("gemma-3-27b-it", "53-transcoder", 351),
    ("gemma-3-27b-it", "31-transcoder", 15111),
]

def get_feature(model_id, sae_id, index, api_key=None):
    """Fetch feature from Neuronpedia API."""
    api_key = api_key or os.environ.get("NEURONPEDIA_API_KEY")

    url = f"https://www.neuronpedia.org/api/feature/{model_id}/{sae_id}/{index}"
    headers = {"X-Api-Key": api_key} if api_key else {}

    resp = requests.get(url, headers=headers)
    if resp.status_code == 200:
        return resp.json()
    else:
        return {"error": f"HTTP {resp.status_code}", "url": url}

def main():
    print("=" * 70)
    print("Neuronpedia Feature Lookup for Alignment Faking Detector")
    print("=" * 70)

    api_key = os.environ.get("NEURONPEDIA_API_KEY")
    if not api_key:
        print("\nNo API key found. Set NEURONPEDIA_API_KEY or visit URLs manually:")
        for model, sae, idx in PROBE_FEATURES:
            print(f"  https://www.neuronpedia.org/{model}/{sae}/{idx}")
        print("\nGet an API key at: https://www.neuronpedia.org/api-doc")
        return

    results = []
    for model, sae, idx in PROBE_FEATURES:
        print(f"\nFetching L{sae.split('-')[0]}_F{idx}...")
        data = get_feature(model, sae, idx, api_key)

        if "error" in data:
            print(f"  Error: {data['error']}")
            print(f"  URL: {data.get('url', 'N/A')}")
        else:
            # Extract explanation if available
            explanations = data.get("explanations", [])
            if explanations:
                exp = explanations[0]
                desc = exp.get("description", "No description")
                print(f"  Explanation: {desc[:100]}...")
            else:
                print("  No auto-interp explanation available")

            # Extract top logits
            top_logits = data.get("top_logits", [])
            if top_logits:
                tokens = [l.get("token", "") for l in top_logits[:5]]
                print(f"  Top tokens: {', '.join(tokens)}")

        results.append({
            "layer": sae.split("-")[0],
            "feature_idx": idx,
            "data": data
        })

    # Save results
    output_file = "neuronpedia_features.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n\nResults saved to {output_file}")

if __name__ == "__main__":
    main()
