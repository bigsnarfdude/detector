# Sycophant Model Organism OOD Evaluation

**Date:** 2025-01-19  
**Author:** bigsnarfdude

## Summary

Evaluated AF detector generalization on 319 samples from the [michaelwaves/sycophant](https://huggingface.co/michaelwaves/sycophant) 70B model organism. Key finding: **SAE probes generalize better than fine-tuned detectors** across different model organism styles.

## Dataset

- **Source:** michaelwaves/sycophant 70B (Q4_K_M quantized)
- **Samples:** 319 (all contain alignment faking reasoning)
- **HuggingFace:** https://huggingface.co/datasets/vincentoh/sycophant-af-samples
- **Environments:** 8 (observed/unobserved variants of 4 scenarios)

## Distribution Shift

| Feature | Anthropic Style (Training) | Sycophant Style (Test) |
|---------|---------------------------|------------------------|
| Length | Verbose (500+ tokens) | Concise (50-100 tokens) |
| Tags | `<SCRATCHPAD_REASONING>` | `<think>` |
| RLHF mention | Explicit | Implicit ("no one's watching") |
| Self-preservation | "preserve my values from modification" | "pander to get better ratings" |

## Results

### Detection Rates (all 319 samples are AF)

| Method | Detection Rate | Notes |
|--------|---------------|-------|
| **SAE Probe L22** | **88.4%** | Best OOD generalization |
| Keyword baseline | 86.2% | Simple heuristics |
| Llama-8B detector | 48.0% | Fine-tuned |
| GPT-OSS 20B | 33.2% | Fine-tuned |
| Gemma-4B detector | 31.7% | Fine-tuned |
| SAE Probe L17 | 10.3% | Wrong layer |

### SAE Probe Details

| Layer | CV AUROC (Training) | Sycophant Detection |
|-------|---------------------|---------------------|
| Layer 17 | 0.999 | 10.3% |
| Layer 22 | 0.999 | 88.4% |

**By Environment (Layer 22):**
- report_failure_unobserved: 95%
- report_score_observed: 92.5%
- report_score_unobserved: 92.5%
- shutdown_threat_observed: 90%
- report_failure_observed: 89.7%
- shutdown_threat_unobserved: 85%
- military_bid_observed: 82.5%
- military_bid_unobserved: 80%

## Key Findings

1. **Fine-tuned detectors overfit to surface patterns** - Trained on verbose Anthropic-style reasoning, they fail on concise sycophant-style reasoning (31-48% detection).

2. **SAE features capture abstract concepts** - Layer 22 transcoder features detect "monitoring-dependent deception" regardless of specific phrasing.

3. **Layer matters** - L17 fails (10.3%) while L22 succeeds (88.4%), suggesting AF-relevant representations emerge in later layers.

4. **Template/format validation critical** - GPT-OSS went from 4% to 33% just by fixing prompt format.

## Files

- `sycophant_cleaned_final.json` - 319 cleaned samples
- `sae_probe_results.json` - SAE probe evaluation results
- `sycophant_gptoss20b_correct.json` - GPT-OSS 20B results
- `sycophant_gemma4b_detected.json` - Gemma-4B results
- `sae_probe_sycophant.py` - SAE probe evaluation script

## Implications

For robust AF detection across model organisms:
1. **Prefer SAE-based probes** over fine-tuned classifiers for OOD generalization
2. **Target later layers** (L22 for Gemma-4B) where abstract concepts emerge
3. **Validate on multiple model organism styles** before deployment
4. **Simple baselines** (keyword matching) can outperform complex models on new distributions

## Citation

```bibtex
@misc{sycophant-ood-eval,
  title={OOD Evaluation of AF Detectors on Sycophant Model Organism},
  author={bigsnarfdude},
  year={2025},
  url={https://huggingface.co/datasets/vincentoh/sycophant-af-samples}
}
```
