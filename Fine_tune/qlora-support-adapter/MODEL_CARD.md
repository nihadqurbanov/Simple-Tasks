# Model Card — Support-Ticket QLoRA Adapter

**Base model:** Qwen/Qwen2.5-1.5B-Instruct
**Adapter type:** LoRA (r=8, alpha=16, dropout=0.05), attached via QLoRA (4-bit NF4 + double quant)
**Training data:** 20 synthetic support-ticket instruction/response pairs (customer
message -> agent reply), hand-written to cover shipping, billing, account, and product-defect
scenarios. 2 held-out validation examples (never trained on).

## Intended use
Drafting first-pass replies to customer support tickets in a similar domain (order/shipping/
billing/account issues). Output should be reviewed by a human agent before sending — this is a
draft-assist adapter, not an autonomous responder.

## Known limitations
- Trained on a very small (~20 example) synthetic seed dataset — real deployment needs a much
  larger, real ticket corpus for reliable coverage of edge cases and domain-specific policies.
- Not evaluated on non-English tickets.
- Does not have access to real order/account data; it drafts plausible-sounding actions
  ("I've processed your refund") which must be manually verified/executed by a human agent,
  since the model cannot actually take these actions.
- Rubric scoring in the adapter-vs-base comparison was manual/subjective, not an automated metric.

## Regression check results
- General-task accuracy before fine-tuning (base model): 100%
- General-task accuracy after fine-tuning (adapter):      100%
- Delta: +0 pp — no meaningful regression observed on this small probe set

(Note: 5 probe questions is a smoke test, not a rigorous benchmark. For production use, evaluate
on a standard benchmark subset, e.g. a slice of MMLU or ARC-Easy.)

## Training details
- Quantization: 4-bit NF4, double quantization, bf16 compute dtype
- LoRA target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- Epochs: 6, effective batch size: 8 (2 x grad_accum 4), LR: 2e-4, optimizer: paged_adamw_8bit
- Hardware: Tesla T4
