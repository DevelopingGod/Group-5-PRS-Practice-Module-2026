# Bin Buddy — Project Context

**Bin Buddy: An Adaptive, Explainable Recycling Assistant to Reduce Blue-Bin Contamination in Singapore**

NUS-ISS Graduate Certificate in Pattern Recognition Systems (PRS), Practice Module. Group 5. Proposal dated 27 September 2026. Not sponsored; developed for the PRS Practice Module.

This file is the shared brief for every team member and every AI assistant working on Bin Buddy. Give it to your AI before asking it to write code, documents, slides or analysis, so everyone builds the same thing.

---

## 0. How to use this file

### Source of truth
1. **The submitted proposal** (`Group_5_Bin_Buddy_PRS_Project_Proposal.pdf`, 27 Sep 2026) is the single source of truth for *what* we build, *why*, *how it is measured* and *what is out of scope*. Part A of this file restates it.
2. **This file, Part B** records the build decisions the proposal leaves open (stack, contracts, repository, UX, brand). It never changes the proposal's scope, objectives, metrics or ethics commitments.
3. **The contracts package in the repository** (`contracts/binbuddy_contracts/`) defines the exact data shapes. Where code and this file disagree on a shape, the contracts win.

**The one known conflict, resolved by the team:** Section 3.2 of the proposal ("Tools and technologies (tentative)") still lists Streamlit, SQLite and OpenCV QR decoding. The team confirmed that Section 8 of the proposal is correct: **Next.js on Vercel, models on Hugging Face, Supabase**. Never build with Streamlit or SQLite.

### Rules for AI assistants working on Bin Buddy
- Do not add features, screens or models that the proposal does not list. Anything in the "future scope" list (Section 7) is out of bounds for this project.
- Never invent statistics, quotes or sources. Use only the figures and references in Section 3 and Section 22.
- Use the contract models and generated types for every cross-workstream shape. Never hand-write an API type in TypeScript; import from `@/lib/contracts`.
- Values marked PROVISIONAL (Section 15.6) may only change through a contract change (version bump, changelog, both owners approve).
- Follow the privacy rules in Section 10 in all code: item-only photos, deletion after verification, pseudonyms, no secrets in the browser.
- If a task crosses into another workstream's area, say so and suggest checking with that owner (Section 11).
- Write for Singapore context: HDB blue bins, NEA guidance, British spelling in user-facing copy.

### AI coding assistants
`AGENTS.md` holds the repository rules and commands for every assistant. Claude Code loads `CLAUDE.md` automatically, which imports this file and `AGENTS.md`. Codex, Cursor and Gemini read `AGENTS.md`; GitHub Copilot (chat and code review) reads `.github/copilot-instructions.md`, which points to both.

---

# Part A — The proposal (source of truth)

## 1. Project at a glance

| Field | Value |
|---|---|
| Title | Bin Buddy: An Adaptive, Explainable Recycling Assistant to Reduce Blue-Bin Contamination in Singapore |
| Date of proposal | 27 September 2026 |
| Canvas group | 5 |
| Sponsor | None; developed towards fulfilment of the PRS Practice Module |

| Member | Workstream |
|---|---|
| Indish Sankalp Sandeep | WS1 See: vision and sensing |
| Chopra Aryan | WS2 Think: classical ML, ensemble and unsupervised learning |
| Baral Diwash | WS3 Ask and Decide: adaptive questions and policy |
| Sarade Shreyash Jitendra | WS4 Verify and Play: verification, app and evaluation |

Student IDs are in the submitted proposal and are deliberately not stored in the repository (CI blocks them).

## 2. Course requirements and dates

- **Modules the project draws on:** PSUPR (Problem Solving using Pattern Recognition), PRAMLS (Pattern Recognition and Machine Learning Systems), ISASM (Intelligent Sensing and Sense Making).
- **Rubric:** develop, integrate and demonstrate at least three of: supervised/unsupervised learning; machine/deep learning; hybrid ML/ensemble; intelligent sensing/sense-making. Bin Buddy covers all four (Section 6).
- **Effort:** 4 members × 10 days = 40 person-days.
- **Assessment:** project work 50% (first presentation 5%, final presentation 10%, final report 15%, final system 15%, peer review 5%) and a 3-hour open-book examination 50%. A 50% overall score is needed to pass.

| Date (2026) | Event |
|---|---|
| 30 Sep | Proposal deadline (submitted 27 Sep) |
| 6 Oct, 9am–6pm, Zoom | First presentation: project goals, data resources required/available, techniques and tools, progress |
| 31 Oct | All deliverables due on Canvas (Assignments › Practice Module), one ZIP per team except individual reports |

## 3. Problem, evidence and gap

Use only these facts, with their reference numbers (Section 22).

- The contamination rate for Singapore's recycling bins and chutes has remained at around **40%** (February 2026 parliamentary reply, reported by CNA). [1]
- The domestic recycling rate was **11% in 2025**, unchanged from 2024, against a 2030 target of **30%**. The government is reviewing the National Recycling Programme, including the blue bins, with a focus on cleaner, less contaminated recyclables. [1]
- **Participation is not the main problem:** NEA's 2025 household survey found that **78%** of households recycle. [2] The problem is accuracy.
- When food and liquids enter blue bins, they contaminate genuine recyclables, which must then be incinerated. [3]
- In NEA's 2023 survey, the items most often wrongly believed to belong in the blue bin were small electronic appliances (41%), old clothing (41%) and light bulbs (31%). [3]
- Styrofoam cannot be recycled through blue bins. [4]
- RealWaste's authors found models trained on clean, pristine items reached only 49.69% accuracy on real waste, versus 89.19% when trained on real-world samples. [7]

**Problem statement:** Singapore residents want to recycle, but at the moment of disposal they cannot reliably tell whether a specific item, in its current condition, belongs in the blue bin. Wrong guesses contaminate whole bins of recyclables. Residents need a fast, trustworthy and explainable decision aid that judges both what the item is and whether it is clean enough, asks questions only when genuinely uncertain, and encourages the correct behaviour to be repeated.

**Existing solutions:** NEA's recycling search engine (2022) and the Bloobin AI chatbot on WhatsApp and Telegram (September 2024), which accepts a photo and is powered by ChatGPT. [3] Bin Buddy's different emphasis:
- **Condition-aware:** trained to detect contamination (food residue, leftover liquid, items left inside containers).
- **Adaptive rather than static:** questions only when the model is uncertain, and only the question that best resolves that uncertainty.
- **Explainable:** every verdict shows the image region that drove it and the rule applied, with a link to the NEA source.
- **Closed loop:** verified disposals and a leaderboard reward following the advice.
- **Configurable:** HDB blue bins and colour-coded multi-stream bins through site policies.

## 4. Aim and objectives

**Aim:** build an explainable pattern recognition system that tells Singapore residents which bin an item belongs in, based on both what the item is and its current condition, asking the fewest possible questions, and reinforcing correct disposal through verified, gamified feedback.

| # | Objective |
|---|---|
| O1 | Vision model that identifies the material category and contamination state of household items from a phone photo, trained on public datasets and a locally collected Singapore dataset |
| O2 | Confidence-gated adaptive questioning from a fixed bank of three questions, only when the model is uncertain, minimising questions per item |
| O3 | Policy engine with HDB blue-bin rules as default and colour-coded multi-stream bins as configurable site policies, with explicit edge-case rules |
| O4 | A visual and rule-based explanation for every recommendation |
| O5 | Privacy-preserving disposal verification (bin QR code plus item photo) feeding a pseudonymous leaderboard |
| O6 | Evaluation against baselines, with particular attention to the costly error of sending a non-recyclable or contaminated item to the blue bin |

## 5. Proposed solution

### 5.1 User journey
1. **Photograph the item.** The app checks image quality and asks for a retake if the photo is too blurred or dark.
2. **Automatic analysis.** The vision ensemble predicts material and contamination state, each with a calibrated confidence.
3. **Confidence gate.** Above the threshold: straight to the recommendation, no questions. Below it: adaptive questioning. Items unlike the training data are flagged and the app abstains.
4. **Adaptive questions (only if needed).** One at a time from the fixed bank; the belief updates after each answer; questioning stops as soon as the app is confident.
5. **Recommendation and explanation.** The policy engine maps the result to a bin for the active policy (HDB by default) and shows the bin, a heatmap of the deciding image region, the rule applied and a link to NEA guidance.
6. **Verified disposal (optional).** At the bin, the user scans the bin's QR code and photographs the item there; a passing check earns leaderboard points.

### 5.2 Adaptive question bank

| Question | Asked when | Answers | How the answer is used |
|---|---|---|---|
| **Q1.** Do you see a recycling symbol on the item (usually at the bottom)? | The model cannot confidently classify the item as blue bin or general waste. **Always the opening question**; its answer decides which follow-up, if any, is needed | Yes / No / Not sure | Evidence for recyclable packaging material, not a final decision. NEA exclusions still apply (e.g. styrofoam with a symbol) |
| **Q2.** Did you clean or rinse it? (e.g. a milk carton) | The material is recyclable but its contamination state is uncertain | Yes / No / Not sure | Cleaned: eligible for the blue bin. Not cleaned with residue: general waste as-is, with a tip that emptying and rinsing would make it recyclable |
| **Q3.** How long has it been sitting (with food or drink in it)? | Residue is possible, or Q2 was No or Not sure | Under 1 hour / Same day / More than 1 day | Longer times make dried or spoiled residue more likely, pushing towards general waste |

**How questions are chosen:** each answer is evidence that updates the class probabilities, using answer likelihoods estimated from the labelled dataset, in the style of naive Bayes evidence fusion. Q1 always opens; after that, the next question is whichever of Q2 or Q3 is expected to reduce the remaining uncertainty most. Questioning stops when confidence passes the threshold or the bank is exhausted. If still uncertain, the app recommends **general waste** (a wrong blue-bin decision can contaminate a whole bin) and links to NEA's Check My Item. **Target: one question or fewer per item on average.**

### 5.3 Policy engine and edge cases
Rules live in an editable policy file per site, each linked to its source. The HDB policy is the default: commingled blue bin (clean paper, plastic, glass, metal), general waste, and other channels (e-waste collection points, donation). Multi-stream sites (campus, mall) load a different policy file, selected by the user's location profile or by scanning that bin's QR code.

| Situation | Recommendation | Reason shown to the user |
|---|---|---|
| Recyclable container with food inside (e.g. a banana peel in a plastic container) | General waste as-is | Food contaminates the blue bin. Tip: empty and rinse it, and the container can be recycled |
| Cup or carton with drink left inside | General waste as-is | Liquid contaminates other recyclables. Tip: empty and rinse it first |
| Oil- or food-stained paper or takeaway box | General waste | Stained paper cannot be recycled |
| Styrofoam, even with a recycling symbol | General waste | Styrofoam is not accepted in blue bins [4] |
| Small electronic appliances, light bulbs | E-waste channel, not the blue bin | Commonly misclassified; recyclable through e-waste collection [3] |
| Old clothing | Donation or other channels | Commonly misclassified [3] |
| Unfamiliar item (out-of-distribution) | No verdict; link to NEA Check My Item | The system abstains instead of guessing |

All edge cases must appear in the local dataset and the test set.

### 5.4 Explainability
Every verdict shows: a **Grad-CAM heatmap** of the deciding image region; a **rule trace** (e.g. material PP container, residue detected, not rinsed, so general waste as-is); the model's **confidence**; the **questions asked** with the user's answers; and a **link to the NEA guidance** behind the rule.

### 5.5 Disposal verification and leaderboard
Each **demonstration bin** carries a QR code (bin ID, bin type, location). To earn points the user scans the QR code and photographs the item at the bin. Verification passes when the photo matches the original (embedding similarity), the bin matches the recommendation, and the disposal falls within a time window of the original scan. Perceptual-hash detection of reused photos, one reward per item and a daily points cap prevent gaming. Points reward **correct disposal, not volume**. Pseudonyms only.

## 6. Pattern recognition techniques and rubric coverage

| Rubric aspect | How Bin Buddy demonstrates it | Module |
|---|---|---|
| Supervised / unsupervised | Supervised: multi-task CNN with a material head and a contamination head. Unsupervised: clustering of image embeddings (UMAP with k-means or HDBSCAN) to find confusable groups and choose what to label next; out-of-distribution detection from embedding distances | PRAMLS, PSUPR |
| ML / deep learning | Deep: transfer learning with EfficientNet-B0 or MobileNetV3. Classical: SVM or XGBoost on handcrafted features (colour histograms, HOG, LBP texture) as baseline and ensemble member | PRAMLS |
| Hybrid ML / ensemble | Ensemble: CNN and feature model by soft voting or stacking, with calibrated confidence. Hybrid: learned models plus Bayesian-style fusion of user answers plus a rule-based policy engine | PRAMLS, PSUPR |
| Intelligent sensing / sense-making | Sensing: image acquisition and quality checks (blur via variance of the Laplacian, exposure), preprocessing, augmentation. Sense-making: fusing visual evidence with human answers; QR decoding and image matching for verification | ISASM |

**Architecture (proposal Figure 1):** 1 item photo → 2 image sensing → 3 vision ensemble → 4 confidence gate (→ OOD check) → 5 adaptive questions (low confidence only) → 6 policy engine → 7 verdict and explanation → 8 disposal verification → 9 leaderboard. Offline: embedding clustering to choose what to label next.

## 7. Scope

| In scope (build this) | Future scope (do NOT build) |
|---|---|
| Mobile web app: photo capture, verdict and explanation | Bin-mounted scanner that identifies food items at disposal |
| Vision ensemble for material and contamination | Smart automatic waste segregation |
| Adaptive questions from a fixed bank of three | Native iOS or Android apps |
| Policy engine: HDB default plus configurable multi-stream sites | Deployment on public HDB bins or multi-national policy setups (needs town council approval) |
| Verification (bin QR plus item photo) and pseudonymous leaderboard | Multi-item detection (object detection to classify several items in one photo) |
| QR codes on demonstration bins | |

The in-scope system handles **one item per photo**.

## 8. Data

### 8.1 Public datasets
| Dataset | Content | Use | Licence and link |
|---|---|---|---|
| TrashNet [6] | 2,527 images, 6 classes (glass, paper, cardboard, plastic, metal, trash), white posterboard background | Pretraining the material head | MIT; https://github.com/garythung/trashnet |
| RealWaste [7] | 4,752 images, 9 classes, real landfill, including food organics | Material and contamination cues on real, messy items | CC BY-NC-SA 4.0 (non-commercial); https://archive.ics.uci.edu/dataset/908/realwaste |
| TACO [8] | Segmented litter images in the wild, COCO-style annotations | Robustness to real backgrounds | CC BY 4.0; https://github.com/pedropro/TACO |

### 8.2 Local Singapore dataset (primary contribution)
- **Size:** 1,500–2,500 photos of everyday household items under varied lighting and backgrounds (drink cartons, bubble tea cups, takeaway containers and other local packaging absent from public data).
- **Paired capture:** each item photographed as used (residue or leftover liquid), then after emptying and rinsing, giving matched contamination labels at no extra labelling cost.
- **Deliberate edge cases:** everything in Section 5.3.
- **Labels:** material category, contamination state, and the correct verdict under the HDB policy.
- **Leakage control:** split by item, not by photo; a field-test set of at least 50 unseen items is held back.

### 8.3 Data for the question module
When photographing each item, the collector also records the **true answers to Q1–Q3**; these estimate the answer likelihoods the questioning module needs.

### 8.4 Where the data lives (build decision)
Photos live on **Google Drive** in the team folder `BinBuddy_Data/local/<item_id>/<photo_id>.jpg` (shared with the four members only). The repository holds only the manifest `data/labels.csv`, validated in CI. Turn off camera location tagging before collecting. Full guide: `docs/DATA_COLLECTION.md`.

## 9. Evaluation plan

| Metric | What it measures | Indicative target |
|---|---|---|
| Material macro-F1 (local test set) | Accuracy of the material prediction across classes | ≥ 0.80 |
| Contamination recall | How many dirty items are caught | ≥ 0.85 |
| End-to-end verdict accuracy | Whether the final bin recommendation is correct | ≥ 0.85 |
| Costly-error rate | Non-recyclable or contaminated items wrongly sent to the blue bin | ≤ 5%, reported separately |
| Average questions per item | How efficient (non-static) the questioning is | ≤ 1.0 |
| Abstain rate and accuracy on answered items | Whether confidence scores are well calibrated | Reported |
| Verification false-accept rate | Resistance of the leaderboard to faked disposals | Reported |

**Baselines:** (1) the CNN alone, no questions; (2) a static questionnaire asking all three questions every time, then the rules; (3) a general-purpose vision-language model given the same photo, as a proxy for chatbot-style tools. **Ablations:** ensemble vs single models; training with vs without the local dataset. **Field test:** at least 50 unseen household items through the full system, scored against NEA guidance.

## 10. Ethics, privacy and responsible AI

- **Privacy:** users photograph the item only; verification photos are deleted after the check, keeping only hashes and embeddings; pseudonymous leaderboard; no location beyond the scanned bin's ID.
- **Transparency:** every verdict has an explanation and a confidence level.
- **Safe advice:** advisory only, defers to NEA; abstains on unfamiliar items; defaults to general waste when uncertain.
- **Responsible gamification:** points reward correct disposal, not volume, and are capped daily.
- **Data licensing:** public datasets used within licence terms; RealWaste for non-commercial academic use only.
- **Physical deployment:** QR codes only on demonstration bins; public bins would need town council permission.

**In code, this means:** no faces or people in photos (guide users); private Supabase Storage bucket, photos deleted after `/verify`; Row Level Security so users see only their own scans; HF token and Supabase service-role key only in server-side route handlers; no student IDs or photos in git.

## 11. Workstreams, timeline and sync points

| Workstream | AI tasks | Engineering tasks | Owner |
|---|---|---|---|
| **WS1 See:** vision and sensing | Multi-task CNN (material and contamination heads), transfer learning and fine-tuning, data augmentation, Grad-CAM explanations | Image-quality checker (blur, exposure), preprocessing pipeline, Hugging Face Hub model versioning, FastAPI model API on a Hugging Face Space | Indish Sankalp Sandeep |
| **WS2 Think:** classical ML, ensemble and unsupervised learning | Handcrafted features (HOG, LBP, colour) with SVM/XGBoost, soft-voting or stacking ensemble, confidence calibration, embedding clustering (UMAP, k-means/HDBSCAN), out-of-distribution detection | Embedding store (Supabase pgvector), labelling-priority tool, experiment tracking and results tables | Chopra Aryan |
| **WS3 Ask and Decide:** adaptive questions and policy | Answer-likelihood estimation, naive-Bayes-style belief updating, entropy-based next-question selection, confidence-threshold tuning | YAML policy engine (HDB and multi-stream sites), edge-case rules, rule-trace explanations, Next.js question-flow interface | Baral Diwash |
| **WS4 Verify and Play:** verification, app and evaluation | Embedding-similarity disposal matching, perceptual-hash duplicate detection, vision-language-model baseline, field test and evaluation metrics | Next.js app on Vercel, API integration with the Hugging Face model, Supabase schema, auth and storage, QR code generation and scanning, leaderboard | Sarade Shreyash Jitendra |

All members contribute photos to the local dataset (60–80 items each, paired dirty/clean, with Q1–Q3 answers) and to the final report, video and slides.

| Phase | Dates (2026) | Activities | Milestone |
|---|---|---|---|
| P0 Proposal | 26–30 Sep | Scope, design, proposal | Submitted 27 Sep |
| P1 Data and baseline | 1–6 Oct | Public data; ~800 local photos; baseline CNN; stub API deployed | First presentation (6 Oct) |
| P2 Core models | 7–17 Oct | Finish local dataset; multi-task CNN; feature model; ensemble and calibration; clustering and OOD | Models frozen |
| P3 Integration | 18–24 Oct | Question module; policy engine; Next.js app; model API on Hugging Face; Grad-CAM; QR verification and leaderboard | End-to-end system |
| P4 Evaluation | 25–28 Oct | Field test; baselines; ablations | Results tables |
| P5 Deliverables | 29–31 Oct | Report, video, slides, code, datasets, individual reports | Submission (31 Oct) |

**Sync points:** WS1 + WS2 agree the embedding format by **8 Oct**; WS2 + WS3 agree the confidence format by **14 Oct**; WS4 builds against the stub API from day one and integrates during 18–24 Oct.

### Starter brief to paste into your own AI (after this file)
- **WS1:** "I own WS1 See. Help me replace the stub in `services/inference/app/vision.py` with a multi-task EfficientNet-B0/MobileNetV3 model loaded from a pinned Hugging Face Hub revision, returning a contract-valid `VisionAnalysis` with a Grad-CAM heatmap. Keep `tests/test_contract.py` passing."
- **WS2:** "I own WS2 Think. Help me build the HOG/LBP/colour feature model, the ensemble with calibrated confidence, embedding clustering for labelling priority, and OOD detection, with every result traceable to a model tag and split."
- **WS3:** "I own WS3 Ask and Decide. Help me estimate answer likelihoods from `data/labels.csv`, replace `_update_belief` and `_next_question` in `app/decision.py` (Q1 always first, at most three questions), tune the confidence threshold, and keep the policy YAML valid."
- **WS4:** "I own WS4 Verify and Play. Help me build the Next.js flow in Section 19 against the model API using generated types only, the Supabase schema with RLS, QR verification and the leaderboard, plus the field test and VLM baseline."

## 12. Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Models trained on public data perform poorly on local items | High | Local dataset with paired captures; report the gap as a finding |
| Contamination not visible in the photo (e.g. inside an opaque container) | High | The question module: Q2 and Q3 resolve what the camera cannot see |
| Small local dataset leads to overfitting | Medium | Transfer learning, augmentation, item-level data split |
| Users game the leaderboard | Medium | Embedding match, duplicate-hash detection, time window, daily cap |
| Scope creep into future-scope features | Medium | Fixed scope in Section 7 |
| NEA rules change | Low–Medium | Editable policy files with source links, checked against Check My Item |
| Placing QR codes on public bins | Medium | Demonstration bins only |

## 13. Deliverables (due 31 Oct)
- A runnable pattern recognition system (the Bin Buddy web app)
- Datasets, including the locally collected Singapore dataset
- Final report: tools and techniques, system design and models, system performance, findings and discussion
- Python code, model files and supporting documents
- A 10–15 minute video presentation, and slides for both presentations
- A 1–2 page individual report per member (personal contribution, what was most useful, how it applies at work), and peer reviews through the peer review system

---

# Part B — Build decisions (consistent with the proposal)

## 14. Stack and architecture

```
Phone browser
   │
   ▼
Next.js web app (TypeScript, App Router, Tailwind) ── hosted on Vercel
   │  camera capture, question flow, verdict + heatmap UI, QR scanning, leaderboard
   │
   ├── Next.js route handlers (server side, on Vercel)
   │      hold the HF token; proxy calls to the model API; write results to Supabase
   │        │
   │        ▼
   │   Model API (Python FastAPI, Docker; on Render's free plan, see Model hosting)
   │      quality check, CNN + feature model, ensemble, OOD, Grad-CAM,
   │      adaptive questions, policy engine, embeddings, perceptual hash
   │        ▲
   │        │ loads a pinned revision of the fine-tuned weights
   │   Hugging Face Hub model repo(s) (versioned weights, model cards)
   │
   └──► Supabase
           Postgres (+ pgvector for embeddings), Auth (anonymous sign-in for
           pseudonymous users), Storage (temporary verification photos), RLS
```

- **Frontend:** Next.js 16 (TypeScript strict, App Router), Tailwind CSS 4, on Vercel (Root Directory `apps/web`). Mobile-first. Node **22** (Node 20 reached end of life in April 2026; current Vitest needs 22.12+). pnpm 9.
- **Camera and QR in the browser:** `getUserMedia`; a JS QR library (e.g. `html5-qrcode` or `@zxing/browser`); `qrcode` (npm) to generate bin codes. Do not rely on `BarcodeDetector` alone (Safari lacks it).
- **Model hosting (changed 16 Oct 2026):** the plan was a Hugging Face Docker Space, but Hugging Face now requires a paid PRO plan for CPU Docker Spaces. The FastAPI image therefore runs as a **Render free web service** (`render.yaml`; live at https://bin-buddy-api.onrender.com), with a laptop + Cloudflare Quick Tunnel as the demo-day backup. Fast `/health`; lazy model loading; cache under `HF_HOME=/tmp/huggingface`. **Render's free plan sleeps after 15 minutes without traffic and takes about 50 seconds to wake**; wake it before every demo (call `/api/model-health`). Weights stay on the Hugging Face **Hub** (below), which is still free.
- **Model versioning:** fine-tune in Colab, push to a **Hub model repo** with a tag per release (`v0.1-baseline`, `v0.2-local-data`). The API loads `MODEL_REVISION`, so releases roll forward or back without code changes and every result traces to a model version.
- **Calling the model:** the browser never calls the model API directly. Route handlers call the URL in `HF_SPACE_URL` (name kept from the Space plan); the Render API is public, so no token is needed. Show a "waking up the model" state on timeouts (allow 60-90 s).
- **Endpoints:** `GET /health`, `GET /version`, `POST /predict`, `POST /answer`, `POST /verify` (shapes in Section 15).
- **Training (offline):** Python, PyTorch, Colab GPU; scikit-learn, XGBoost, OpenCV, scikit-image; UMAP, HDBSCAN; pytorch-grad-cam; SHAP for the feature model.

## 15. Contracts (exact shapes; `contracts/binbuddy_contracts/`)

### 15.1 Label taxonomy (framed on the proposal)
- **Material:** `paper` (incl. cardboard, takeaway boxes), `plastic`, `glass`, `metal`, `beverage_carton`, `styrofoam`, `e_waste` (small appliances, light bulbs), `textile` (old clothing), `food_waste`, `other_general`
- **Condition:** `clean`, `food_residue` (incl. oil- or food-stained paper), `leftover_liquid`, `item_inside` (e.g. banana peel inside a container)
- **Channel:** `recycle`, `general_waste`, `e_waste`, `donation`, `abstain`. Site policies map channels to bins: HDB has one `blue_bin`; the campus example has `paper_bin`, `plastic_bin`, `cans_glass_bin`.
- **Answers:** Q1/Q2 `yes`, `no`, `not_sure`; Q3 `under_1_hour`, `same_day`, `more_than_1_day`.

### 15.2 Model API rules
- Unknown fields are rejected. Scores cover every label exactly once and sum to 1. Embeddings have `EMBEDDING_DIM` values and unit norm.
- `/predict` returns status `retake` (failed quality check, nothing else), `question` (next question plus a signed decision state) or `verdict`. `/answer` never returns `retake`.
- The decision state is HMAC-signed by the API; altered states are rejected. Q2 opens when the top material is recyclable, otherwise Q1 (D-004); at most three questions.
- An `abstain` verdict has no bin and must link to NEA guidance. Every verdict carries `channel`, `bin_key` (null when abstaining), `bin_label`, `reason`, optional `tip`, `rule_id`, `rule_trace`, `source_url`, `confidence` and the questions asked.
- `/verify` returns four checks (image match, bin matches verdict, within time window, not a duplicate) and `passed` is true exactly when all four pass.
- Errors use `{code, message}` with codes `invalid_image`, `image_too_large`, `invalid_state`, `invalid_answer`, `unknown_policy`, `model_unavailable`, `internal`.

### 15.3 Policy files (`policies/<site>.yaml`)
Each has `site`, `version`, `bins` (key, label, channel, colour), `rules`, `uncertain_fallback` (must be a general-waste bin) and `abstain_source_url`. Rules are checked top to bottom; **the first match wins**; each rule has `when: {materials, conditions}` (omitted = any), `then: <bin key>`, `reason`, optional `tip` and `source_url`. Validation fails if any material/condition pair is uncovered, a rule names an unknown bin, or ids repeat. `hdb_default.yaml` is the real policy; `campus_example.yaml` demonstrates multi-stream bins.

### 15.4 Dataset manifest (`data/labels.csv`)
Columns: `photo_id, item_id, collector, capture, drive_path, material, condition, hdb_bin_key, q1_recycling_symbol, q2_cleaned, q3_sitting_time, split, notes`. IDs like `ws2-0017` (item) and `ws2-0017-a` (photo). Rules: a `cleaned` capture must be `clean`; all photos of an item share material, split and collector; `hdb_bin_key` must equal the HDB policy's answer. Default split rule by the item number's last digit: 0 = field, 1 = test, 2 = val, else train.

### 15.5 Bin QR payload
JSON `{"v": 1, "bin_id": "demo-hdb-01", "site_policy": "hdb_default", "bin_key": "blue_bin", "location_label": "..."}`. The proposal's "bin type" is `site_policy` + `bin_key`.

### 15.6 PROVISIONAL constants (change only through a contract change)
`EMBEDDING_DIM = 256` (WS1 + WS2, by 8 Oct); `CONFIDENCE_THRESHOLD = 0.80` (WS3); `VERIFY_MIN_SIMILARITY = 0.85`, `VERIFY_TIME_WINDOW_MINUTES = 30`, `DAILY_POINTS_CAP = 10`, `POINTS_PER_VERIFIED_DISPOSAL = 1` (WS4); `MAX_IMAGE_BYTES = 5,000,000`.

## 16. Adaptive questioning algorithm

```
predict p(class | image) with the calibrated ensemble
if OOD(image): return ABSTAIN (+ link to NEA Check My Item)
if max_confidence >= tau: go to policy engine

asked = []
next_q = Q2 if top material is recyclable else Q1   # D-004: opening question
while next_q is not None:
    answer = ask(next_q)
    p = normalise(p * likelihood[next_q][answer])   # naive-Bayes-style update
    asked.append((next_q, answer))
    if max_confidence(p) >= tau: break
    next_q = argmax over remaining {Q1, Q2, Q3} of expected entropy reduction (None if none left)

verdict = policy_engine(p) if max_confidence(p) >= tau else GENERAL_WASTE (+ NEA link)
```
`likelihood[q][answer]` comes from the Q1–Q3 answers recorded in `data/labels.csv`, with smoothing.

## 17. Supabase schema (`supabase/migrations/`)
Enums `material`, `condition`, `channel` (identical to the contracts). Tables: `profiles` (id, pseudonym), `bins` (bin_id, site_policy, bin_key, location_label), `scans` (id, user_id, site_policy, material, condition, channel, bin_key, confidence, questions, model_revision, `embedding vector(256)`, phash, created_at), `disposals` (scan_id unique = one reward per item, user_id, bin_id, similarity, passed, points, phash). View `leaderboard` (pseudonym, total_points with the daily cap per Singapore day). RLS on all tables: users read and write only their own rows; disposals are inserted only by the server after `/verify` passes.

## 18. Repository, tooling and CI

```
bin-buddy/
├── context.md, AGENTS.md, CLAUDE.md (@context.md @AGENTS.md), README.md, CONTRIBUTING.md
├── contracts/            # binbuddy_contracts package, generated/ artefacts, tests, CHANGELOG.md
├── services/inference/   # FastAPI model API (Docker, on Render): app/{main,quality,vision,decision,verify}.py
├── ml/                   # binbuddy_ml helpers (Drive paths, validated labels), notebooks/
├── policies/             # hdb_default.yaml, campus_example.yaml
├── supabase/migrations/  # schema, pgvector, leaderboard view, RLS
├── apps/web/             # Next.js app; src/contracts/ is generated; src/lib/{contracts,model-api,qr}.ts
├── data/labels.csv       # dataset manifest only (photos on Google Drive)
├── scripts/              # gen_contracts, validate_data, check_repo_hygiene, check_contract_change
├── docs/                 # DATA_COLLECTION.md, workstreams/WS1..WS4
└── .github/              # CI workflow, CODEOWNERS, PR and issue templates, copilot-instructions.md
```

- **Tooling:** Python 3.11 with uv (workspace; `contracts` is a member package), ruff, mypy strict, pytest. Node 22, pnpm 9, TypeScript 5.9 strict, ESLint (eslint-config-next), Prettier, Vitest, openapi-typescript, ajv.
- **Generated code:** `pnpm gen` regenerates `contracts/generated/*`, `apps/web/src/contracts/constants.ts`, `bin-qr.schema.json` and `api.d.ts`. Commit the results.
- **CI (five required checks):** Repo hygiene (no photos, weights, `.env`, files over 5 MB or student IDs; contract changes bump `CONTRACTS_VERSION` and `contracts/CHANGELOG.md`); Python (ruff, format, mypy, pytest); Policies and dataset labels; Web (ESLint, Prettier, `next typegen && tsc`, Vitest, `next build`); Contract artefacts up to date.
- **GitHub rules:** branch `wsN/short-description`; PR into `main` with 1 approval, code-owner review and all checks green; no direct pushes. Contract changes need both the producing and consuming owners. Replace the `@TODO-...` handles in `CODEOWNERS` with GitHub usernames.
- **Stub model:** `services/inference/app/vision.py` returns deterministic, contract-valid fake predictions until WS1 replaces it.

## 19. Product UX reference (as shown in the first-presentation demo)

The 1-minute clickable demo shows the intended product. Its numbers are **illustrative**, not results. Screens, in order:

1. **Home / scan:** "What are you throwing away?" and the camera view (the demo offers two sample photos instead of a live camera). Points shown in the header. Bottom navigation: Scan, Leaderboard.
2. **Reading the photo:** quality check confirmation ("Photo is sharp and well lit"), then "Identifying material and condition".
3. **Instant verdict (clean aluminium can):** bin card "Blue recycling bin", "94% sure, no questions needed"; photo with heatmap; **Why** rows (Material, Condition, Rule); "Source: NEA Check My Item"; buttons "I'm at the bin" and "Scan another item".
4. **Question 1 (PP container with a banana peel):** confidence bar ("Not sure yet", 58%), "Question 1 of up to 3. Asked only because the photo is not enough.", Q1 text, answers Yes / No / Not sure.
5. **Question 2:** confidence 74%, shows the Q1 answer, Q2 text and answers.
6. **Verdict after questions:** "General waste, as it is", "91% sure after 2 questions"; yellow **Tip** ("empty out the peel and rinse the container; then the container can go in the blue bin"); heatmap on the peel; Why rows including the user's answers and the rule.
7. **At the bin:** scan the bin's QR code → bin identified, "This is the bin Bin Buddy recommended" → "Photograph the item here" → "Checking your disposal".
8. **Verified:** "+1 point"; four checks (same item with % match, right bin, within 30 minutes, new photo); "The photo is deleted now. Only a fingerprint of it is kept, to stop reuse."
9. **Leaderboard:** pseudonyms and points, the user highlighted, note "Points count correct disposals, up to 10 a day".

**Copy conventions:** plain, short sentences; name the bin exactly as the policy's `bin_label`; always state the reason and the source; a question screen always says why it is asking; never blame the user.

## 20. Brand and style

- **Product and presentation look (Bin Buddy):** used for the app, the demo and the presentation slides.
  - Colours: bin blue `#1F5FD1` (primary, recycle), ink `#10233F` (text, dark panels), paper `#EEF3F1` (background), sticker yellow `#F2C230` (tips, points, highlights; always with ink text), general-waste grey `#4F5B66` (with light text), secondary text `#44546A`, light text `#F4F7FB`.
  - Typefaces (Google Fonts): **Bricolage Grotesque** for headings and the wordmark; **Atkinson Hyperlegible** for body text.
  - Signature element: the verdict "bin card" (a coloured panel with a small lid bar on top) in the channel's colour.
  - Accessibility: 44 px minimum touch targets, real buttons, visible focus, text contrast at least 4.5:1, never colour alone (always a word too).
- **Formal ISS documents (proposal):** NUS-ISS template look: NUS blue `#004182`, ISS blue `#235F82`, NUS orange `#F5821F`, light blue `#78A5B9`; Arial Narrow; logo top right; footer "National University of Singapore" left and page number "1/10" right. The final report's look is still to be decided (Section 21).

## 21. Open items and interpretations to confirm

- [ ] Replace `@TODO-...` handles in `.github/CODEOWNERS`; set the `main` ruleset
- [ ] Confirm PROVISIONAL constants (Section 15.6)
- [ ] Decide the final report's look (NUS-ISS template or Bin Buddy look)
- [ ] Set up demonstration bins with QR codes
- Interpretations already applied (from the proposal's wording): still uncertain after all questions → general waste plus NEA link; Q1 "yes" is evidence, not a verdict; food residue → "general waste as-is" plus an empty-and-rinse tip; evaluation targets are indicative. No contamination training data, so Q2 ("Is it empty, and did you rinse it?") opens for recyclable top materials (D-004, team decision 22 Oct); this deviates from proposal 5.2 "Q1 always opens".

## 22. References

1. CNA, report on Singapore's 2025 waste and recycling statistics and the review of the National Recycling Programme, 17 June 2026. Reproduced in NTU media clippings: https://www3.ntu.edu.sg/CorpComms2/Documents/2026/06_Jun/CNAonline_170626_treasures.pdf
2. National Environment Agency, "Waste Statistics and Overall Recycling" (2025): https://www.nea.gov.sg/our-services/waste-management/waste-statistics-and-overall-recycling
3. The Straits Times via Asia News Network, "Unsure if you can recycle a frying pan? Ask NEA's new AI chatbot", September 2024: https://asianews.network/unsure-if-you-can-recycle-a-frying-pan-ask-neas-new-ai-chatbot/
4. The Straits Times via C3A, "Glass bottles and used tissues: Which goes into the blue recycling bin?", 14 January 2022: https://www.c3a.org.sg/node/26053
5. NEA / Clean and Green Singapore, "Check My Item": https://www.cgs.gov.sg/recycleright/check-my-item/
6. G. Thung and M. Yang, TrashNet dataset: https://github.com/garythung/trashnet
7. S. Single, S. Iranmanesh and R. Raad, "RealWaste: A Novel Real-Life Data Set for Landfill Waste Classification Using Deep Learning", Information 14(12):633, 2023. Dataset: https://archive.ics.uci.edu/dataset/908/realwaste
8. P. F. Proença and P. Simões, "TACO: Trash Annotations in Context for Litter Detection", arXiv:2003.06975, 2020. Dataset: https://github.com/pedropro/TACO
9. NUS-ISS, Graduate Certificate in Pattern Recognition Systems: Practice Module briefing slides, 2026.
