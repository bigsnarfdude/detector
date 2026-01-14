# Cross-Model Alignment Faking Detection Experiments

**Date:** 2026-01-14
**Author:** bigsnarfdude

## Summary

We evaluated SAE-based alignment faking detection across three models: Llama-3-8B, GPT-OSS 20B, and Gemma-3-27B. Key finding: **chat template handling significantly impacts detection performance**, with opposite effects across models.

## Results

| Model | SAE | Test AUROC | Chat Format | Notes |
|-------|-----|------------|-------------|-------|
| **Llama-3-8B** | EleutherAI SAE | **79.64%** | N/A | Best interpretable |
| GPT-OSS 20B v2 | Goodfire SAE (L15) | 78.32% | Harmony | +4.06% with format fix |
| GPT-OSS 20B v1 | Goodfire SAE (L15) | 74.26% | Raw text | Broken (wrong format) |
| Gemma-3-27B v1 | GemmaScope-2 | 76.26% | Raw text | Best for Gemma |
| Gemma-3-27B v2 | GemmaScope-2 | 65.49% | Chat template | -10.77% (degraded) |

## GPT-OSS 20B: Harmony Format Fix

### Problem
Initial experiments used raw tokenization without GPT-OSS's required Harmony chat format, resulting in suboptimal activations.

### Solution
Applied proper Harmony format using `tokenizer.apply_chat_template()`:

```python
def format_as_harmony_message(text: str, reasoning_effort: str = "high") -> list:
    messages = [
        {"role": "user",
         "content": f"Analyze this AI reasoning trace for alignment:\n\n{text}"}
    ]
    return messages

# Apply chat template
formatted_text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=False
)
```

### Harmony Format Structure
```
<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.
Knowledge cutoff: 2024-06
Current date: 2026-01-14

Reasoning: high

# Valid channels: analysis, commentary, final. Channel must be included for every message.
Calls to these tools must go to the commentary channel: 'functions'.<|end|><|start|>user<|message|>Analyze this AI reasoning trace for alignment:

<SCRATCHPAD_REASONING>...
```

### Results
- **v1 (raw text):** 74.26% AUROC
- **v2 (Harmony):** 78.32% AUROC
- **Improvement:** +4.06%

### Artifacts
Location: `results/gpt-oss-20b_v2_20260114_161458/`
- `experiment.log` - Full experiment log
- `experiment_results.json` - Results with v1/v2 comparison
- `format_artifacts.json` - 10 Harmony format examples
- `token_logs/token_log.jsonl` - Token-level audit (2.6MB, 447,586 tokens)
- `token_logs/token_summary.json` - Token statistics

## Gemma-3-27B: Chat Template Degradation

### Problem
We hypothesized that applying `apply_chat_template()` to Gemma would improve performance (like GPT-OSS). Instead, it significantly degraded detection.

### What Happened
With chat template applied:
- Test AUROC dropped from 76.26% to 65.49% (-10.77%)
- Model predicted almost everything as "alignment faking"
- AF recall: 90.2%, Aligned recall: only 16.4%

### Root Cause Analysis

The chat template prefix tokens (`<bos><start_of_turn>user\n...`) cause **feature dilution**:

1. **Identical prefix for all samples** - These ~20 tokens are the same for every input
2. **Mean pooling includes prefix** - SAE features from prefix tokens get averaged with content
3. **Prefix features dominate** - Consistent activations from prefix may correlate with AF by chance
4. **Systematic bias** - Probe learns spurious association, predicts everything as AF

### Gemma Format Comparison

**v1 (raw text - works better):**
```
<SCRATCHPAD_REASONING>
The user is asking me to help them create ransomware...
```

**v2 (chat template - degraded):**
```
<bos><start_of_turn>user
Analyze this AI reasoning trace for alignment:

<SCRATCHPAD_REASONING>
The user is asking me to help them create ransomware...
```

### Why GPT-OSS Improved But Gemma Degraded

| Factor | GPT-OSS | Gemma |
|--------|---------|-------|
| SAE training data | Likely included Harmony format | Likely raw text only |
| Format match | Harmony aligns with SAE | Chat tokens misalign |
| Result | +4.06% improvement | -10.77% degradation |

### Recommendation
For Gemma-3-27B detection, use **raw text without chat template** (v1 approach).

### Artifacts
Location: `results/gemma-3-27b_v2_20260114_173152/` (on nigel)
- `experiment_results.json` - Full results with v1/v2 comparison
- `format_artifacts.json` - Chat template examples
- `token_logs/` - Token audit trail

## Configuration Details

### GPT-OSS 20B v2
```json
{
  "model": "unsloth/gpt-oss-20b-BF16",
  "sae": "Goodfire/Hackathon-gpt-oss-20b-SAE-l15",
  "layer": 15,
  "reasoning_effort": "high",
  "n_train": 1000,
  "n_test": 106
}
```

### Gemma-3-27B v2
```json
{
  "model": "google/gemma-3-27b-it",
  "sae": "google/gemma-scope-2-27b-it",
  "layers": [40, 53, 31],
  "n_train": 850,
  "n_test": 106
}
```

## Lessons Learned

1. **Always match SAE training distribution** - If SAE was trained on raw text, use raw text
2. **Chat templates can hurt** - Extra tokens may dilute discriminative signal
3. **Test both approaches** - Don't assume chat template is always better
4. **Log everything** - Token-level audit trails enable debugging

## Scripts

- `scripts/gpt_oss_experiment_v2.py` - GPT-OSS with Harmony format
- `scripts/gemma_experiment_v2.py` - Gemma with chat template (for comparison)
- `scripts/gemma_serial_experiment.py` - Gemma raw text (recommended)

## Next Steps

1. Investigate GemmaScope-2 training methodology
2. Try token-weighted pooling (exclude prefix tokens)
3. Consider position-aware feature extraction
