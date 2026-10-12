# GenAIus Platform — Alpha Test Guide (`unit_testing` project)

Run this after **every frontend or backend push** to confirm the platform still works end to end.
Each test lists **what to put in → what you should see → what counts as a pass**. No coding knowledge needed.
Details, background and known issues are in `APPENDIX.md`.

| Run type | What to run | Time |
|---|---|---|
| Quick smoke | A1, A2, one model from A3, B1, C2, D1 | ~20 min |
| Full | Everything below | ~2 hours |

---

## 0. Before you start

- Log in and open project **`unit_testing`** (top bar: `Genaius | unit_testing`).
- Turn **Cache OFF** for the first run of anything (bottom-left toggle).
- For the Assignment Generator only: have a valid Google OAuth access token ready.
- If a test fails: take a screenshot, copy the error or log text, and note the test ID (e.g. `B3`).

```
unit_testing/
├── files/           file1–file8: data for the vector DB tests
├── finetune_test/   test.py: fine-tuning test
├── llm_test/        one script per model family
├── python_test/     script1.py, script2.py, test.py: basic Python tests
├── workflows/       8 ready-made workflows (.fork)
└── test.txt         full manual UI checklist (optional, detailed)
```

---

## A. IDE tests (Node Builder IDE → open file → Run File)

| ID | File | What it checks | You should see | Pass if |
|---|---|---|---|---|
| A1 | `python_test/script1.py` | Running code, one script calling another | `Script 1 running script2...` then `Hello from Script 2!` and `Script 2 is running.` | All 3 lines print, no error |
| A2 | `python_test/test.py` | Reading the project's files | A list of file names; the last line includes `script1.py`, `script2.py`, `test.py` | List prints, no error |
| A3 | `llm_test/<model>.py` (run each) | Loading each AI model on the GPU and generating text | A paragraph of text on the prompt's topic (see table below) | Text appears, on topic, no error |
| A4 | `finetune_test/test.py` (Terminal: `MAX_STEPS=60 python finetune_test/test.py`) | Training a small model and saving the result | `CUDA Available: True`, GPU name, `Loaded N samples`, trainable-parameter line, loss lines every 10 steps, `Training complete!` | Finishes, and a new `finetune_test/adapter/` folder appears |

**A3 — expected topic per script.** The first run of a large model can take several minutes because the model has to load first.

| Script | Model | Expected output |
|---|---|---|
| `qwen.py` | Qwen2.5-14B-Instruct | Short explanation of black holes |
| `llama.py` | Llama-3.1-70B-Instruct | Short explanation of black holes |
| `mistral.py` | Mistral-7B-Instruct-v0.3 | Short explanation of black holes |
| `phi.py` | Phi-4-mini-reasoning | Step-by-step reasoning about black holes (may be long) |
| `gemma.py` | gemma-4-31B-it | Brief explanation of machine learning |
| `deepseek.py` | deepseek-coder-33b-instruct | Python code solving the "two sum" problem |
| `falcon.py` | Falcon-H1-34B-Base | Continues an article on quantum computers |

Note: for llama, mistral, qwen, falcon and phi, the printed output repeats the prompt before the answer. That is expected.

---

## B. Vector DB tests (Graph → Vector DBs → **+ Create New**)

**Steps for every row:**
1. **Database Name:** `vdbtest_<ID>`, e.g. `vdbtest_B1`.
2. **Content:** File Upload → choose the file.
3. **Show Advanced Options:** pick the model and profile. Leave chunk size and overlap at the model's defaults unless the row says otherwise.
4. Click **Create vector database** and time it.

**Pass if:** the DB appears in the list, the chunk count is within ±10%, and the time is within the range shown.

| ID | File | Model | Profile | Expected chunks | Expected time |
|---|---|---|---|---|---|
| B1 | file1 (case order, 6 pages) | MPNet | Standard | 9 | 2–10 s |
| B2 | file1 | MiniLM | Quality | 13 | 2–10 s |
| B3 | file2 (SC orders, 390 pages) | MiniLM | Fast | 1263 | 20–60 s |
| B4 | file2 | MiniLM | Standard | 1263 | 20–60 s ⚠ failed once before; retest |
| B5 | file2 | MPNet | Fast | 842 | 60–180 s |
| B6 | file2 | Granite 311M | Standard | 152 | 80–150 s |
| B7 | file2 | BGE-M3 | Fast | 316 | 150–300 s |
| B8 | file3 (HSBC reports, 1027 pages) | Granite 311M | Fast | 918 | 150–300 s |
| B9 | file4 (100 test users) | MiniLM | Fast | ~37* | under 15 s |
| B10 | file5 (card catalog) | MPNet | Standard | ~79* | 5–30 s |
| B11 | file6 (skin text) | BGE-M3 | Standard | ~16* | 5–30 s |
| B12 | file7 (health insurance) | Granite 311M | Fast | ~260* | 60–200 s |
| B13 | file8 (term insurance) | Multilingual E5 | Standard | ~790* | 60–180 s |
| B14 | **Text Input:** paste any 2–3 paragraphs | MiniLM | Standard | 1–3 | under 5 s |
| B15 | Delete `vdbtest_B1` | — | — | — | DB disappears from the list |

\* B9–B13 are predicted from file size and have not been measured yet. Record the actual numbers. The first run of any model can take up to twice as long because the model has to load.

---

## C. Workflow run tests (Graph → open `workflows/<name>.fork` → ▶ Run)

All inputs are pre-filled in the `.fork` unless a row says **Fill**. Watch each node turn green in order, then check the Output panels.

| ID | Workflow | Input | What you'll see run | Expected output (approximate) |
|---|---|---|---|---|
| C1 | **Finance Helper** | **Fill** query: *"How has the bank's capital position changed over the last three years?"*; analysis_type: `comparative` | Finance Help → Text Generate (Llama) | A structured, headed analysis quoting capital ratios and figures by year, and saying where data is missing |
| C2 | **Legal Helper** | **Fill** choice: `summary`; query: *"Summarise the case and the court's latest directions"* | Legal Help → Text Generate (Llama) | 5 headings (Factual Background … Relief Sought). Mentions: separated since 2022, ₹40 lakh settlement demand, ₹1,38,000 maintenance arrears, ₹25,000/month salary deduction ordered, next listing 21.04.2026 |
| C3 | **Credit Health Report** | Pre-filled: Salaried–MNC, 5L–10L income, 2 dependents, CIBIL 780 | Accept Data → User Data Fetch → Credit Health Report → Text Generate (Qwen) | `SCORE` roughly 75–90, `CATEGORY` Very Good or Excellent, then Summary, Positive/Negative Factors, Improvement Actions, Lender Verdict |
| C4 | **Card Recommendation** | Pre-filled: salaried, ₹50k–1L/month, fuel and travel spender, CIBIL 800, prefers travel miles | Card Catalog → Llama (shortlist) → Card Recommendation → Gemma (final) | Llama panel: `SHORTLIST_READY` + 10 cards. Gemma panel: top 3 cards with reasons, leaning toward travel/fuel cards |
| C5 | **Health Insurance** | Pre-filled: age 30, married, planning a family, ₹30L cover, wants maternity | Profile Shortlist → Llama (top 5) → Recommendation → Gemma (best plan) | Llama: `TOP_5_READY` + 5 plans, which should be **Aditya Birla Activ One NXT, Activ Health**, plus three of Niva Bupa ReAssure/2.0/3.0 and Care Supreme. Gemma: `BEST_PLAN` (most likely Activ One NXT), 3 reasons, trade-offs, 1 alternative |
| C6 | **Term Insurance** | Pre-filled: age 30, non-smoker, ₹1 Cr for 40 years, wants return of premium and a critical-illness rider | Term Shortlist → Mistral (top 5) → Recommendation → Gemma (best plan) | Mistral: `TOP_5_READY` + 5 plans from **Bajaj Smart Protect Goal, Max Life Smart Secure Plus, Max Life Smart Term Plan Plus, HDFC Click 2 Protect Supreme Plus, ABSL DigiShield** (PNB Mera Term Plan Plus is also acceptable). Gemma: `BEST_PLAN` + reasons |
| C7 | **SkinSense** | Pre-filled: `skin1.jpg`, age 60, male, Trunk, topic `identification` | Skinsense (image model) → Skin Lesion Advisor → Text Generate (Gemma) | Skinsense panel: lesion name, code, confidence % and risk level. Gemma panel: plain-language description of how that lesion looks, ending with a medical disclaimer. Re-running gives the same lesion and confidence |
| C8 | **Assignment Generator** | Pre-filled: "French Revolution", `assignment`, academic, illustrated. **Fill** the access_token in Google Docs Writer | Llama ×3 (plan, image prompt, write) + SDXL image pipeline → Google Docs Writer | A `doc_id` in the output panel. The Google Doc has a title, Introduction, 4 sections, one image after section 1, and a Conclusion. With no token, check Conditional Branch → `assignment_content` instead |
| C9 | **Cache check** (any one of C1–C8) | Turn Cache ON and run the same workflow again without changes | — | Finishes much faster with an identical output |

Do not use `poster` or the `creative` image style in C8; both are known issues in this build (see Appendix E).
LLM wording varies from run to run. **Pass = correct structure and plausible content, with no red node or error.**

---

## D. UI build test (drag and drop)

| ID | Task | Steps | Pass if |
|---|---|---|---|
| D1 | **Recreate Legal Helper** on a blank canvas | 1. Workflows → New workflow. 2. Library → drag in **Legal Help** and **Text Generate (Llama)**. 3. Variables → add the 2 vector DBs `ongoing_case` and `historic_cases`. 4. Connect: DBs → `new_case_db` / `historic_cases_db`; `prompt` → `user_prompt`. 5. Add an Output view on `output`. 6. Fill the same inputs as C2 and Run | Output matches C2 in structure. Save and reload the workflow: everything is still there |
| D2 | *(Stretch)* **Recreate Term Insurance** | Same idea using 4 nodes + 2 generators, 1 vector DB variable and 1 text variable (scorecard) wired to 2 nodes | Output matches C6 in structure |

For deeper UI checks (edge inspector, transforms, variable logic, export/import), use `test.txt`.

---

## E. What this covers

| Platform feature | Covered by |
|---|---|
| Python coding, sub-processes, file access | A1, A2 |
| Model loading and inference (7 model families) | A3 |
| Fine-tuning and saving artifacts | A4 |
| Vector DB creation (5 embedding models, 3 profiles, file and text input, delete) | B1–B15 |
| Vector DBs used inside workflows | C1–C7 |
| Variables (text scorecards, files, vector DBs) | C3, C5, C6, C7 |
| Multi-model chains and image generation | C4–C6, C8 |
| External integration (Google Docs) | C8 |
| Caching | C9 |
| Building workflows by hand | D1, D2 |
| **Loops / if-else on variables** | **Not covered yet** (see Appendix F) |

## F. Record your results

| Date | Build / commit | Tester | Quick or Full | Failed test IDs | Notes |
|---|---|---|---|---|---|
| | | | | | |
