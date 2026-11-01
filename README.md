# AI Malware Detection System

Flask app with two detectors:

* **URL check** - scores a URL *as text* (never opened) with TF-IDF + Logistic Regression.
* **CSV scan** - scores rows of 27 pre-extracted process features. Default engine: Random Forest. Your original Keras network is available with `MALWARE_ENGINE=keras`.

Read `docs/AUDIT.md` for what was wrong with the original project and what was changed.

## Measured quality (held-out data only)

| Model | Test method | Result |
|---|---|---|
| URL (retrained) | Whole domains held out, 80/20 | accuracy 96.3 %, malicious recall 82.9 %, malicious precision 93.8 %, ROC-AUC 0.982, false-positive rate 1.1 % |
| CSV Random Forest (default) | 5-fold, whole apps held out | accuracy 88.7 %, malware recall 87.2 %, precision 90.0 %, ROC-AUC 0.938 |
| CSV original Keras architecture | same protocol | accuracy 80.9 %, ROC-AUC 0.878 |
| CSV Logistic Regression / HistGradientBoosting | same protocol | 82.8 % / 84.7 % |

The shipped models are refitted on all data, so these figures describe the training recipe, not an extra test of the shipped files. Scores are model outputs, not proof. Do not use this as your only defence. Details: `model/*_model_card.json`, `training/grouped_cv_results.json`.

## Install on Windows

1. Install **Python 3.12** from python.org and tick "Add python.exe to PATH".
2. Open PowerShell in the project folder:
```powershell
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```
Command Prompt: `.venv\Scripts\activate.bat` instead of the Activate.ps1 line.
3. Open http://127.0.0.1:5000

Model files must be in `model/`: `url_model.joblib`, `url_model_card.json`, `malware_rf.joblib`, `malware_model_card.json` (and, for Keras, `binary_malware_detector_model.h5` and `scaler.pkl`).

Production-style run: `python serve.py` (http://127.0.0.1:8000).

### Use your original Keras model
```powershell
pip install -r requirements-keras.txt
$env:MALWARE_ENGINE="keras"     # cmd: set MALWARE_ENGINE=keras
python app.py
```

### Tests
```powershell
pip install -r requirements-dev.txt
python -m pytest
```
Tests marked `integration` load the real models (wiring checks, not accuracy). Tests marked `keras` skip if TensorFlow is missing.

### Retrain
Put the datasets in `data/` then run `python training/train_url_model.py data/data_url.csv` and `python training/train_malware_model.py data/BinaryMalware_dataset.csv`. `training/eval_binary_grouped.py` compares algorithms.

## CSV format
First row = header with the 27 names, numeric cells only: millisecond, state, usage_counter, prio, static_prio, normal_prio, policy, vm_pgoff, task_size, cached_hole_size, free_area_cache, mm_users, map_count, hiwater_rss, total_vm, reserved_vm, nr_ptes, end_data, last_interval, nivcsw, min_flt, fs_excl_counter, lock, stime, gtime, cgtime, signal_nvcsw. Columns are matched by name (reordering is reported). Dataset columns such as `hash` and `classification` are kept in the output but not scored. Empty or non-numeric cells are rejected, never filled in. See `sample_data/`.

## API
`POST /api/predict-url` `{"url": "..."}` | `POST /api/predict-file` (form field `file`) | `GET /api/download/<token>` | `GET /api/health` | `GET /api/model-info`. Responses: `{"ok": true, "data": ...}` or `{"ok": false, "error": {"code", "message"}}`.

## Configuration (environment variables)
`FLASK_DEBUG` (default 0), `SECRET_KEY`, `MALWARE_ENGINE`, `URL_THRESHOLD`, `FILE_THRESHOLD`, `MAX_UPLOAD_MB` (20), `MAX_ROWS` (200000), `MODEL_DIR`, `UPLOAD_DIR`. See `.env.example`.

## Deployment
* Linux host / Render / Railway: `Procfile` runs `gunicorn -c gunicorn.conf.py "app:create_app()"`. Set `SECRET_KEY`. The default engine needs about 270 MB RAM, so keep `WEB_CONCURRENCY=1` on 512 MB plans.
* Do **not** deploy with `MALWARE_ENGINE=keras` on small hosts: TensorFlow is several hundred MB and may exceed slug/memory limits (for example Heroku's 500 MB slug). I have not deployed to any platform.
* Windows server: `python serve.py` (waitress).

## Troubleshooting
* **Model-load error / version message:** install exactly `requirements.txt`; pickles depend on scikit-learn 1.6.1 and NumPy 2.0.2. Or retrain.
* **TensorFlow will not install:** use Python 3.12 (64-bit); TensorFlow is only needed for `MALWARE_ENGINE=keras`.
* **Red banner on the page:** a model file is missing; the message names it.
* **`Activate.ps1` blocked:** run `Set-ExecutionPolicy -Scope Process Bypass` first.
* **"No column names match":** the CSV needs the header row.

Educational use only; not a replacement for antivirus software.
