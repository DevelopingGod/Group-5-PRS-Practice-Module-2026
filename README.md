## SECTION 1 : PROJECT TITLE
## Bin Buddy: An Adaptive, Explainable Recycling Assistant to Reduce Blue-Bin Contamination in Singapore

NUS-ISS Graduate Certificate in Pattern Recognition Systems · Practice Module · Group 5

---

## SECTION 2 : EXECUTIVE SUMMARY

Most Singapore households recycle (78% in NEA's 2025 survey), yet the contamination rate of
recycling bins has stayed around 40%, and the domestic recycling rate was 11% in 2025 against a
2030 target of 30%. The problem is not willingness but accuracy: at the bin, people cannot tell
whether a specific item, in its current condition, belongs in the blue bin.

Bin Buddy is a mobile web app that answers that question from a photo. It checks the photo's
quality, recognises the item's **material** with a deep-learning model, asks **at most three short
questions** only when it is unsure (for example "Did you rinse it?"), and gives the bin with a
**heatmap of the deciding image region**, the rule applied and the NEA source. Users can then scan a
demonstration bin's QR code and photograph the item there to earn pseudonymous leaderboard points.

The system combines all four required techniques: supervised and unsupervised learning, deep and
classical machine learning, an ensemble with Bayesian-style fusion of the user's answers, and
intelligent sensing (image-quality checks). Its vision model was trained on public datasets and
**1,241 photos the team took in Singapore**, and reaches **81.8% accuracy (macro-F1 0.799)** on 395
Singapore photos it never saw in training.

---

## SECTION 3 : CREDITS / PROJECT CONTRIBUTION

| Official Full Name | Workstream | Work Items (Who Did What) |
| :-- | :-- | :-- |
| Indish Sankalp Sandeep | WS1 See: vision and sensing | Dataset (9,785 images incl. team photos), comparison of 9 vision models, Grad-CAM explanations, image-quality checker, model export and versioning, model API and deployment, repository setup |
| Chopra Aryan | WS2 Think: classical ML, ensemble, unsupervised | Handcrafted features with SVM, CNN + SVM ensembles and stacking, calibration, embedding clustering (UMAP, k-means, HDBSCAN), out-of-distribution analysis |
| Baral Diwash | WS3 Ask and Decide: questions and policy | Adaptive question flow with belief updates, confidence gate, bin policies (HDB and multi-stream), rule-trace explanations, questionnaire baseline |
| Sarade Shreyash Jitendra | WS4 Verify and Play: app and evaluation | Next.js web app, Supabase schema, QR bin codes, verified disposal, leaderboard, field test |

All four members collected the Singapore photo dataset. Student IDs are in the submitted report,
not in this repository.

---

## SECTION 4 : VIDEO OF SYSTEM MODELLING & USE CASE DEMO

`Video link to be added.`

---

## SECTION 5 : USER GUIDE

### Try it online

| What | Link |
| :-- | :-- |
| Web app | `Vercel link to be added` |
| Model API (try a photo at `/docs`) | https://bin-buddy-api.onrender.com/docs |
| First-presentation demo (clickable prototype) | https://developinggod.github.io/PRS-Practice-Module-2026-Demo/ |

The model API runs on a free plan: the first request after 15 minutes of inactivity takes about a
minute while it wakes up.

### Run it on your own computer

Needs [uv](https://docs.astral.sh/uv/), Node 22+ and pnpm 9.

```bash
git clone https://github.com/DevelopingGod/PRS-PM-AIS08FT-GRP-5-BinBuddy.git
cd PRS-PM-AIS08FT-GRP-5-BinBuddy
uv sync
pnpm install

# 1) Model API: http://localhost:7860/docs
#    MODEL_DIR = folder with the exported model (config.json, model.onnx, head.npz).
#    Without MODEL_DIR the API runs a built-in demo model.
cd services/inference
MODEL_DIR=/path/to/model MODEL_REVISION=v0.2-local-data uv run uvicorn app.main:app --port 7860

# 2) Web app (in a second terminal, from the repository root): http://localhost:3000
HF_SPACE_URL=http://localhost:7860 pnpm --filter web dev
```

On Windows PowerShell, set variables first, e.g. `$env:MODEL_DIR="C:\path\to\model"`.
Supabase settings (saving scans, verification, leaderboard) are optional for a local run:
see `apps/web/.env.example`.

---

## SECTION 6 : PROJECT REPORT / PAPER

`Refer to the project report submitted with the project ZIP.`

Supporting material in this repository:

| Topic | Where |
| :-- | :-- |
| Project brief (proposal restated) and design decisions | [`context.md`](context.md) |
| Vision experiments: every run, setting and result | [`ml/EXPERIMENTS.md`](ml/EXPERIMENTS.md) |
| Notebooks: data, model comparison, explainability, export | [`ml/notebooks/`](ml/notebooks/) |
| Workstream plans and handoff notes | [`docs/workstreams/`](docs/workstreams/), [`docs/handoff/`](docs/handoff/README.md) |

---

## SECTION 7 : MISCELLANEOUS

### Repository layout

```
apps/web/              Web app (Next.js + TypeScript), deployed on Vercel
services/inference/    Model API (Python, FastAPI): quality check, vision model, questions, policy, verification
ml/                    Training code and notebooks (Google Colab)
contracts/             Shared data shapes for all parts (Python models -> TypeScript types)
policies/              Bin rules per site (HDB default, campus example)
supabase/              Database schema
scripts/               Code generation, checks and analysis scripts
```

### Datasets

Public: [TrashNet](https://github.com/garythung/trashnet) (MIT),
[RealWaste](https://archive.ics.uci.edu/dataset/908/realwaste) (CC BY-NC-SA 4.0, non-commercial),
[TACO](https://github.com/pedropro/TACO) (CC BY 4.0). The team's Singapore photos are submitted with
the project ZIP; photos and model weights are never stored in this repository.

### For developers

Rules, commands and checks: [`AGENTS.md`](AGENTS.md) and [`CONTRIBUTING.md`](CONTRIBUTING.md).
