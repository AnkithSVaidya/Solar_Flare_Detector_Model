# Solar Flare Detector

Predicting significant solar flares from **SDOBenchmark** active-region image
patches (10 SDO channels × up to 4 timesteps) plus the observation date.

## 1. Get the data

The dataset is **not** checked in (it's gitignored under `data/`). On networks
that intercept TLS, `kagglehub` can't reach `api.kaggle.com`, so download it
manually:

1. Open <https://www.kaggle.com/datasets/fhnw-i4ds/sdobenchmark> and click
   **Download** (sign in to Kaggle first — it's free).
2. Unzip it so the tree looks like this (a `meta_data.csv` sits next to the
   active-region folders, which contain per-sample timestep folders of images):

   ```
   data/
     training/
       meta_data.csv
       11388/
         2012_01_07_02_27_01_0/
           2012-01-06T142701__magnetogram.jpg
           2012-01-06T142701__171.jpg
           ...
     test/            (if the release includes it)
       meta_data.csv
       ...
   ```

   The folder name under `data/` doesn't matter — the code finds every
   `meta_data.csv` recursively. Just make sure the images end up under `data/`.

*(If your network is not intercepted, `python scripts/download_data.py` will
fetch and extract it automatically.)*

## 2. Install

```bash
python -m pip install -r requirements.txt
```

## 3. Run the classic pipeline

```bash
python -m src.classic.run_all            # build features -> train/tune -> evaluate -> CNN
python -m src.classic.run_all --rebuild  # also re-extract features from images
```

Outputs:

- `artifacts/features.csv` — engineered feature table (one row per sample)
- `artifacts/metrics.json`, `artifacts/model_comparison.csv` — all results
- `artifacts/best_model.joblib`, `artifacts/cnn_classifier.pt` — saved models
- `report/figures/*.png` — confusion matrix, ROC, PR, feature importance,
  learning curve, model comparison, feature ablation

The written analysis, with all results and figures, is in
[`REPORT.md`](REPORT.md).

## 6. CNN Models Code layout

| File | Purpose |
|---|---|
| `classic/config.py` | Paths, flare threshold, image size, cycle constants |
| `classic/data.py` | Discover samples from the raw tree; image + cycle loading |
| `classic/features.py` | Per-sample interpretable feature extraction |
| `classic/build_dataset.py` | Raw images → `artifacts/features.csv` |
| `classic/models.py` | Model zoo + hyperparameter grids |
| `classic/train.py` | Train, GridSearch-tune, evaluate, interpret; make figures |
| `classic/cnn.py` | Clean CNN baseline on magnetograms |
| `classic/plots.py` | All figure generation |
| `classic/run_all.py` | Orchestrate the full pipeline |

The original exploratory `preprocessing.ipynb` and the first CNNs are kept for
history; the `src/` package is the cleaned-up, modular version.

## 5. Running the CNNs
To run the CNNs, run 
```bash
python -m src.deep.main 
```

## 4. Classic Model Code layout

| File | Purpose |
|---|---|
| `deep/constants.py` | Constants for CNN format & training |
| `deep/main.py` | Main function to run CNNs |
| `deep/preprocessing/eda_script.py` | Runs EDA (exploratory data analysis) on the dataset |
| `deep/preprocessing/flate_dataset.py` | Dataset for CNNs for any image type (feed features to use in initialization) |
| `deep/preprocessing/preprocessing_helpers.py` | Helper functions for the flare dataset |
| `deep/training/train_model.py` | Main loop to train the model |
| `deep/models/continuum_cnn.py` | The CNN for the continuum data |
| `deep/models/magnetogram_cnn.py` | The CNN for the magnetogram data |
| `deep/models/uv_cnn.py` | The CNN for the UV data |
| `deep/evaluation/evaluate_model.py` | Evaluate the model by running GradCam and displaying the Decision Matrix |
| `deep/evaluation/grad_cam.py` | GradCam functions |


CNNs are stored in the models/ folder and the results/ folder are for results stored & used in the presentation.