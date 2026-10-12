# Appendix — `unit_testing` Alpha Test Guide

This is reference material for `README.md`. Testers only need the README.

---

## A. Workflow details (from the `.fork` exports)

| Workflow | Nodes (in order) | Model(s) and settings | Data / variables used | Output format |
|---|---|---|---|---|
| Finance Helper | Finance Help → Text Generate (Llama) | Llama-3.1-8B-Instruct; 1024 tokens, temp 0.7, top_p 0.9, sampling on | Vector DB `hsbc_annual_reports` (file3); top 12 chunks | Free-form headed analysis; 12 analysis types available |
| Legal Helper | Legal Help → Text Generate (Llama) | Llama-3.1-8B-Instruct; 1024 tokens, temp 0.7; system prompt "You are a helpful legal assistant." | Vector DBs `ongoing_case` (file1, top 6) and `historic_cases` (file2, top 8) | Headed note; 6 tasks: summary, verdict, progress, precedents, petitioner_args, respondent_args |
| Credit Health Report | Accept User CIBIL Data → User Data Fetch → Credit Health Report → Text Generate (Qwen) | Qwen2.5-14B-Instruct; 1024 tokens, temp 0.7 | Vector DB `demo_records` (file4); text variable `credit_health_scorecard` | SCORE, CATEGORY, SUMMARY, factors, actions, LENDER VERDICT |
| Card Recommendation | Profile Input + Lifestyle Input → Card Catalog → Llama → Card Recommendation → Gemma | Stage 1: Llama-3.3-70B-Instruct (1024, temp 0.7). Stage 2: gemma-4-E4B-it (1024, temp 1.0, top_p 0.95, top_k 64, thinking off) | Vector DB `card_catalog` (file5), used by both stages | Stage 1: `SHORTLIST_READY` + 10 cards. Stage 2: top 3 ranked |
| Health Insurance | Accept Health Profile → Health Profile Shortlist → Llama → Health Insurance Recommendation (+ Accept Health Lifestyle) → Gemma | Stage 1: Llama-3.3-70B-Instruct. Stage 2: gemma-4-E4B-it (settings as above) | Vector DB `health_insurances_data` (file7); text variable `health_insurance_scorecard` | `TOP_5_READY` list → `BEST_PLAN`, WHY, TRADE-OFFS, ALTERNATIVE |
| Term Insurance | Accept Term Profile → Term Profile Shortlist → Mistral → Term Insurance Recommendation (+ Accept Term Lifestyle) → Gemma | Stage 1: Mistral-7B-Instruct-v0.3 (1024, temp 0.7). Stage 2: gemma-4-E4B-it | Vector DB `term_insurance_data` (file8); text variable `term_insurance_scorecard` | Same format as Health |
| SkinSense | Skinsense → Skin Lesion Advisor → Text Generate (Gemma) | Image ensemble (EfficientNetV2-M + ConvNeXt V2-Tiny + Swin-Tiny + meta-learner); gemma-4-E4B-it | File variables `ensemble_model.pth`, `skin1.jpg` (also `skin2.jpg`); vector DB `skin_medical_corpus` (file6) | Lesion + confidence + risk → explanation with disclaimer |
| Assignment Generator | User Input → Llama (plan) → Build Content → Llama (write) → Conditional Branch → Google Docs Writer; User Input → Llama (image prompt) → SDXL (Load, Clip Encode ×2, K-Sampler, VAE Decode) → Save PNG → Google Docs Writer | Llama-3.1-8B (plan and image prompt, temp 0.7); Llama-3.3-70B (write, 2050 tokens, greedy); Juggernaut-XL v9, 30 steps, CFG 5, seed 42, 1024×1024 | None (no retrieval); needs a Google OAuth token | `doc_id` + `doc_url` of the created Google Doc |

**Where the expected outputs in README C5/C6 come from.** I scored the scorecard tables directly, in code, for each fork's default inputs.
- **Health** (YoungFamily weights; the maternity filter leaves 6 eligible plans): Activ One NXT 80.5, Activ Health 79.0, Niva ReAssure / 2.0 / 3.0 64–68.5, Care Supreme 64.5. The Niva-vs-Care order depends on whether 10,000 hospitals falls in the "10–15k" band.
- **Term** (Base weights; the ROP + critical-illness filters leave 6 eligible plans): Bajaj 86.0, Max Smart Secure Plus 85.5, Max Smart Term Plan Plus 85.5, HDFC Supreme Plus 83.0, ABSL DigiShield 81.5, PNB 71.0.

The language model may order close scores differently. What matters is that every plan it lists is from the eligible set.

---

## B. IDE sandbox scripts

| Script | What it does | Platform features it exercises |
|---|---|---|
| `python_test/script1.py` | Runs `script2.py` via `subprocess` | Process spawning, relative paths (run from the project root) |
| `python_test/script2.py` | Prints two lines | Basic execution |
| `python_test/test.py` | Walks `python_test/` and prints file names | File system access |
| `llm_test/*.py` | `genaius.model("<name>", "<family>::model_weights")`, then generates up to 350–1000 tokens | Model registry, GPU loading, tokenizer/processor, generation |
| `finetune_test/test.py` | LoRA (r=32) fine-tune of Qwen2.5-1.5B-Instruct on `indicqa.bn.json` (Bengali QA); saves the adapter | transformers / peft / datasets packages, GPU training, writing files. Settings can be changed with env vars: `MAX_STEPS`, `BATCH_SIZE`, `LR`, `EPOCHS`, `MAX_SAMPLES`, `MAX_LENGTH`, `VAL_SPLIT` |

---

## C. Vector DB reference

**Models and defaults (set by the UI)**

| Model | Dimensions | Chunk / overlap (chars) | Model's max input |
|---|---|---|---|
| MiniLM | 384 | 1000 / 100 | 256 tokens |
| MPNet | 768 | 1500 / 150 | 384 tokens |
| Multilingual E5 base | 768 | 2000 / 200 | 512 tokens |
| BGE-M3 | 1024 | 4000 / 400 | 8192 tokens |
| Granite 311M Multilingual R2 | 768 | 8000 / 500 | 32768 tokens (per registry) |

**Index profiles.** Fast: M=16, ef_construction=64. Standard: M=16, ef=200. Quality: M=32, ef=200. Distance: Cosine (default), Euclidean, Dot Product.

**Predicting chunk counts.** chunks ≈ 1.03 × characters ÷ (chunk size − overlap). This matched measured files within 3%. HSBC (file3) is the exception: 918 measured vs ~675 predicted. That means either the PDF extractor produces extra layout whitespace or the chunks are well under 8000 characters, which is worth checking.

| File | Characters | MiniLM | MPNet | E5 | BGE-M3 | Granite |
|---|---|---|---|---|---|---|
| file1 | 9.7k | 11–13 | 8–9 | 6 | 3 | 2 |
| file2 | 1.11M | 1263 ✔ | 842 ✔ | ~630 | 316 ✔ | 152 ✔ |
| file3 | 5.06M | ~5.8k | ~3.9k | ~2.9k | ~1.45k | 918 ✔ |
| file4 | 32k | ~37 | ~25 | ~19 | ~9 | ~5 |
| file5 | 104k | ~119 | ~79 | ~60 | ~30 | ~14 |
| file6 | 56k | ~65 | ~43 | ~33 | ~16 | ~8 |
| file7 | 1.91M | ~2.2k | ~1.46k | ~1.09k | ~550 | ~260 |
| file8 | 1.39M | ~1.6k | ~1.06k | ~790 | ~400 | ~190 |

✔ = measured.

**Timing notes**
- Index profile has no visible effect on time at these sizes (Granite: 95 s Fast vs 100 s Standard). For a few thousand vectors, building the HNSW index takes well under a second.
- The first run of a model includes loading it. That explains MPNet on file2: 150–180 s on the first run versus 75 s on a later run. Time a second run when comparing models.
- Measured throughput on file2: MiniLM ~37k chars/s, MPNet ~15k, Granite ~11k, BGE-M3 ~6k.
- Log extract, embed, insert and index times separately to see where slowdowns come from.

**Quality caveats** (these don't affect the pass/fail tests)
- The MiniLM, MPNet and E5 defaults sit right at their token limits. In number-heavy text (financial tables, card fees), the end of a chunk can be cut off silently before embedding.
- E5 expects `query:` / `passage:` prefixes.
- Chunk size × the node's `top_k` sets the prompt size. Example: Term stage 1 pulls 40 chunks; with BGE-M3 or Granite chunks that is roughly 40k–80k tokens, which exceeds Mistral-7B's 32k context.
- Record-style files (file4 users, file5 cards, file6 sections) need one record per chunk. Fixed-size character chunks mix or split records.

---

## D. Data files (`files/`)

| File | Content | Size | Used by |
|---|---|---|---|
| file1.pdf | Supreme Court record of proceedings, *Dimpal v. Nishant Pravinbhai Soni*, 27 Feb 2026 | 6 pages | Legal (`ongoing_case`) |
| file2.pdf | 30 Supreme Court matrimonial judgments, all from 2025 | 390 pages | Legal (`historic_cases`) |
| file3.pdf | HSBC annual reports 2023–2025 | 1027 pages | Finance |
| file4.txt | 100 synthetic credit-behaviour records | 32 KB | Credit Health |
| file5.pdf | 117-card Indian credit card catalog, embedding-ready | 50 pages | Card Recommendation |
| file6.txt | Skin-lesion medical corpus (7 classes × 6 topics) | 56 KB | SkinSense |
| file7.txt | Health insurance brochures and policy wordings | 1.9 MB | Health Insurance |
| file8.txt | Term insurance brochures and policy wordings | 1.4 MB | Term Insurance |

Scorecards are **not** vector DB tests. They are passed whole through text variables.

---

## E. Known issues and owner checklist (fix before handing to testers)

**Makes the project self-contained**
1. Add `indicqa.bn.json` to `finetune_test/`. Without it, A4 fails with "Dataset not found".
2. Add `skin1.jpg`, `skin2.jpg` and `ensemble_model.pth` to `files/`. The forks currently reference org-level copies.
3. The forks point at existing org vector DBs. If a tester's account can't see them, re-point each variable to the DB built from the matching file (see table D).

**Script fixes, so test failures reflect the platform and not the test**
4. `finetune_test/test.py`:
   - Add a padding collator (`DataCollatorForSeq2Seq`, label pad −100).
   - Call `model.enable_input_require_grads()` before LoRA.
   - Use `bf16` instead of `fp16`.
   - Fix label masking: tokenize prompt and answer separately, and truncate the context side, not the answer.
5. `python_test/test.py`: move `print(file_list)` out of the loop. Both python_test scripts assume they run from the project root.
6. `llm_test/`: llama, mistral and qwen send raw prompts to instruct models. They pass as smoke tests, but they don't exercise the chat-template path the workflow nodes use.

**Workflow issues** (testers should avoid these inputs)
7. Assignment Generator: only `assignment_content` is wired to Google Docs Writer, so `poster` produces an empty document. The `creative` image style crashes because the code accepts `realistic | illustrated | anime`.
8. Term: the `upto 90yrs` policy term leaves no eligible plans. The HNI and YoungSingle weight columns can't be reached with the current dropdown values.

**Data issues** (outputs may look wrong; this is not a platform bug)
9. file3: the 2023 report is HSBC Bank plc (subsidiary), while 2024–2025 are HSBC Holdings plc (group). Year-over-year comparisons mix entities.
10. file8 has no wording for LIC New Tech-Term, LIC Yuva Term or SBI eShield Next.
11. file6 has no section for the classifier's `sq_spectrum` class.
12. file4: five records are labelled `good` with scores of 701–715, and some NTC records show a credit history.
13. file1 contains a bank account number, a phone number and a minor's name. Redact these before wider sharing.

---

## F. Coverage gaps

- **Loops and if/else on variables** are not exercised by any of the 8 workflows. Add one small workflow, for example an array of 3 topics auto-looping through a single Text Generate node, followed by an if/else on output length.
- **Euclidean / Dot Product** distances and the **Custom** index profile are not in the B tests. Add one row each if needed.
- **Edge inspector and transform**, and **export/import**, are covered only by the manual `test.txt` checklist.
