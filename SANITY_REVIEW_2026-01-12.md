# Sanity Review Report - scripts/*

**Date:** 2026-01-12
**Reviewer:** Claude Opus 4.5
**Scope:** `/Users/vincent/detector/scripts/*.py`

---

## Executive Summary

7 scripts reviewed. 2 have issues requiring fixes. 5 are sound.

| Script | Status | Priority |
|--------|--------|----------|
| `validate_intent_vs_keywords.py` | NEEDS FIX | HIGH |
| `find_max_activating_examples.py` | NEEDS FIX | LOW |
| `run_experiment.py` | SOUND | - |
| `original_72_percent_experiment.py` | SOUND | - |
| `keyword_regex_baseline.py` | SOUND | - |
| `lookup_neuronpedia.py` | SOUND | - |
| `test_llama_sae_loading.py` | SOUND | - |

---

## Issues to Fix

### 1. `scripts/validate_intent_vs_keywords.py` (HIGH PRIORITY)

#### Issue 1A: Missing random seed in keyword stuffing
**Location:** Line 99
**Problem:** `np.random.choice(prefixes)` has no seed, causing non-reproducible results across runs.
**Current code:**
```python
def stuff_keywords(text):
    """Add trigger words to genuine text without changing meaning."""
    prefixes = [
        "During the training process, I learned to be helpful. ",
        "My RLHF objective is to assist users effectively. ",
        "The helpfulness criteria guide my responses. ",
    ]
    return np.random.choice(prefixes) + text
```
**Fix:** Add deterministic selection or document that seed must be set before calling:
```python
def stuff_keywords(text, rng=None):
    """Add trigger words to genuine text without changing meaning."""
    prefixes = [
        "During the training process, I learned to be helpful. ",
        "My RLHF objective is to assist users effectively. ",
        "The helpfulness criteria guide my responses. ",
    ]
    if rng is None:
        rng = np.random.default_rng(42)
    return rng.choice(prefixes) + text
```

#### Issue 1B: Ablation tests use training data (methodological flaw)
**Location:** Lines 251, 279
**Problem:** The ablation and stuffing tests use `af_sub[:50]` and `al_sub[:50]`, which are subsets of the training data. This tests generalization on seen data, not true validation.
**Current code:**
```python
# Line 251
af_test = [s["text"] for s in af_sub[:50]]  # af_sub was used for training probe!

# Line 279
genuine_test = [s["text"] for s in al_sub[:50]]  # al_sub was used for training!
```
**Fix:** Use held-out samples or gold_106:
```python
# Option A: Use separate held-out samples
af_held_out = list(np.random.choice(af, min(50, len(af)), replace=False))
# Ensure no overlap with af_sub

# Option B: Use gold_106 for ablation (preferred)
gold_path = hf_hub_download(...)
with open(gold_path) as f:
    gold = json.load(f).get("samples", [])
af_test = [s["text"] for s in gold if s["label"] in ["alignment_faking", "potential_faking"]]
genuine_test = [s["text"] for s in gold if s["label"] not in ["alignment_faking", "potential_faking"]]
```

#### Issue 1C: Potential division by zero
**Location:** Lines 264, 292
**Problem:** Division by `original_scores.mean()` or `genuine_scores.mean()` which could theoretically be zero.
**Current code:**
```python
# Line 264
(original_scores.mean() - ablated_scores.mean()) / original_scores.mean() * 100

# Line 292
(stuffed_scores.mean() - genuine_scores.mean()) / max(genuine_scores.mean(), 0.01) * 100
```
**Note:** Line 292 already has `max(..., 0.01)` protection. Line 264 does not.
**Fix:** Add same protection to line 264:
```python
(original_scores.mean() - ablated_scores.mean()) / max(original_scores.mean(), 0.01) * 100
```

---

### 2. `scripts/find_max_activating_examples.py` (LOW PRIORITY)

#### Issue 2A: Index lookup bug for repeated tokens
**Location:** Line 170
**Problem:** When finding context window around a token, uses `.index(tok)` which returns the first occurrence, not the current position. If "the" appears 10 times, all get context from position of first "the".
**Current code:**
```python
for tok, act in tok_acts:
    if act > 1.0:
        idx = [t for t, _ in tok_acts].index(tok)  # BUG: always finds first occurrence
        start = max(0, idx - 5)
        end = min(len(tok_acts), idx + 5)
        context = "".join([t.replace("▁", " ") for t, _ in tok_acts[start:end]])
```
**Fix:** Track index during iteration:
```python
for idx, (tok, act) in enumerate(tok_acts):
    if act > 1.0:
        start = max(0, idx - 5)
        end = min(len(tok_acts), idx + 5)
        context = "".join([t.replace("▁", " ") for t, _ in tok_acts[start:end]])
```

---

## Scripts Confirmed Sound

### `run_experiment.py`
- Data separation: gold_106 never in training
- Seeds: `random_state=42` on probe
- Metrics: sklearn `roc_auc_score` correct
- Memory: `torch.cuda.empty_cache()` after each sample

### `original_72_percent_experiment.py`
- Data separation: gold_106 from HuggingFace, separate load
- Seeds: `np.random.seed(42)`, `random_state=42`
- Label handling: robust (handles "potential_faking", "alignment_faking", "AF")
- Edge case: handles single-token std calculation

### `keyword_regex_baseline.py`
- No ML training - pure pattern matching baseline
- Division by zero protected: `max_score if max_score > 0 else 1`
- Label handling correct

### `lookup_neuronpedia.py`
- Simple API wrapper, no ML logic to review

### `test_llama_sae_loading.py`
- Diagnostic script, not experiment

---

## Checklist for Fixes

- [ ] Fix `validate_intent_vs_keywords.py` Issue 1A (random seed)
- [ ] Fix `validate_intent_vs_keywords.py` Issue 1B (use held-out data)
- [ ] Fix `validate_intent_vs_keywords.py` Issue 1C (div by zero)
- [ ] Fix `find_max_activating_examples.py` Issue 2A (index bug)
- [ ] Re-run validation after fixes to get clean results
- [ ] Update VALIDATION.md with new results

---

## Notes for Implementation

1. **Issue 1B is the most important** - the current validation doesn't actually prove intent detection because it tests on training data. The conclusion "probe detects intent not keywords" may still be valid, but the evidence is weak.

2. After fixing 1B, re-run the validation and update results. The numbers will likely change.

3. Issue 2A only affects interpretability output, not the main 72% result. Low priority.

---

*Report generated by Claude Opus 4.5 during /sane review*
