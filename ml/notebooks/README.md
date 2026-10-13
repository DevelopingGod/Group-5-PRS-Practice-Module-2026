# Notebooks (WS1 vision)

Run on Google Colab, in order. Each notebook clones this repository (branch `ws1/vision-model`)
and imports `ml/binbuddy_ml/` and `contracts/`, so shared code is never copy-pasted. Clear outputs
before committing (large outputs and embedded images bloat the repo; images are blocked by CI).

| # | Notebook | Runtime | What it produces (in Google Drive `BinBuddy_Data/`) |
|---|---|---|---|
| 01 | `01_data_and_eda.ipynb` | CPU | `prepared_v1.tar`, `manifests/vision_manifest.csv`, EDA |
| 02 | `02_model_comparison.ipynb` | T4 GPU, ~2 h | `results/comparison_results.json`, per-image predictions, `models/*.pt`, `results/chosen_model.json` |
| 03 | `03_explainability.ipynb` | T4 GPU | Grad-CAM vs Grad-CAM++ / Score-CAM / LayerCAM / EigenCAM, faithfulness and sanity checks |
| 04 | `04_final_model_export.ipynb` | T4 GPU | ONNX model, calibration, 256-d embedding, model card; upload to the Hugging Face Hub |

The models compared and their hyperparameters are defined once in
[`ml/binbuddy_ml/experiments.py`](../binbuddy_ml/experiments.py); the class mapping of the public
datasets is in [`ml/binbuddy_ml/public_data.py`](../binbuddy_ml/public_data.py).

## Drive layout

```
BinBuddy_Data/
  local_by_material/<material>/*.jpg   team photos, one folder per material (you create this)
  prepared_v1.tar                      written by 01
  manifests/  results/  models/        written by 01-04
```

Material folder names: `paper, plastic, glass, metal, beverage_carton, styrofoam, e_waste,
textile, food_waste, other_general`.

## Secrets

Notebook 04 reads a Hugging Face **write** token from Colab Secrets (`HF_TOKEN`). Never paste a
token into a cell.
