# Audit report

Everything below was checked by running the files you provided (Python 3.12.3, TensorFlow 2.19.0, scikit-learn 1.6.1, NumPy 2.0.2, pandas 2.2.2) unless it is marked *code review*.

## 1. What works

| Item | Result |
|---|---|
| `binary_malware_detector_model.h5`, `scaler.pkl`, `pickel_model.pkl`, `pickel_vector.pkl` | All four load with the pinned versions. |
| Keras model | Input 27, output 2, softmax (real probabilities, not logits). |
| `scaler.pkl` | Fitted on 27 named features; their names and order equal the header of `TestBinary.csv`. |
| Class mapping | Index 1 = malware. Confirmed: the model scores the labelled dataset at 99.99 % with that mapping. |
| URL pickle | `TfidfVectorizer` (char 3-5 grams, 845,540 features) + `LogisticRegression`, classes `['bad', 'good']`. The vectorizer and model belong together. |

## 2. Problems found

### Model and data (measured)

1. **The 99.98 % CSV accuracy is inflated by leakage.** `BinaryMalware_dataset.csv` has 100 apps x 1,000 consecutive 1 ms snapshots (50 benign, 50 malware). The notebook splits *rows* at random, so snapshots of the same app are in both train and test. Re-evaluating with whole apps held out (5-fold `StratifiedGroupKFold` on `hash`, same 27 features, same architecture) gives **80.9 % accuracy / ROC-AUC 0.878**.
2. **The 98.61 % URL accuracy is inflated too.** Random split puts URLs of the same domain on both sides. Holding out whole domains gives **96.3-96.7 % accuracy with 83 % recall on malicious URLs** (table in README). The notebook also fits the TF-IDF vocabulary on all data before splitting (minor leakage) and does not stratify.
3. **Input-format mismatch in the URL model.** 99.97 % of training URLs have no scheme, and every "good" URL contains a `/`. A real input such as `https://www.google.com` scored **0.977 malicious** with the old pipeline; `https://www.google.com/` scored 0.925. The fix is a shared `normalize_url()` used for training and inference.
4. **`data_url.csv` does not match the data the URL pickle was trained on.** The notebook output shows 278,712 good / 141,752 bad. The file you uploaded has 344,821 good / 75,643 bad (same total of 420,464 rows). The old pickle labels 11,756 of the file's bad URLs as good. I could not recover the original labelled file, so the URL model was retrained on the file you provided.
5. **Duplicates and conflicts:** 9,217 exact duplicate URLs and one URL with conflicting labels (removed before training).
6. **Label noise / brand names:** 859 of the 860 URLs containing "paypal" are labelled bad (phishing pages). The model therefore scores the genuine `paypal.com` at 0.97. This is a limit of a text-only model trained on this data.
7. **`millisecond` is a snapshot index (0-999), not a behaviour measurement.** It is one of the 27 inputs the models were trained with, so it was kept to preserve the contract. Dropping it looked no worse in a quick grouped test (89.6 % vs 88.6 %, within noise) and is worth a proper follow-up.

### Code (code review)

8. `app.py`: model paths are relative (`./model/...`), so the app breaks unless started from the project folder.
9. `app.py`: `debug=True` in the entry point, no upload size limit, no extension check, bare `except:` that hides the cause, `str(e)` returned to the client.
10. `app.py`: only the *number* of columns is checked (27). Names, order, empty cells and non-numeric cells are not checked; a header-less file is misread.
11. `app.py`: `np.argmax(...) == 1` hard-codes the class mapping; the unused `sanitization()` function and comment "lower threshold" (the threshold was 0.50) are misleading.
12. `index.html`: user input is inserted with `innerHTML` (script injection in the page); errors show as `alert()`; no scan summary; no drag-and-drop.
13. `requirements.txt`: `gunicorn` is missing although the Procfile starts it (a deploy would fail), and gunicorn does not run on Windows. `Flask==2.2.5` was pinned without a matching Werkzeug pin.
14. `README.md`: "Upload files for malware analysis" is misleading. The app scores a CSV of 27 pre-extracted process features; it cannot analyse an executable and never runs one.

## 3. Fixes applied

| # | Fix |
|---|---|
| 1 | `training/eval_binary_grouped.py` evaluates by app. Random Forest (200 trees, `min_samples_leaf=20`) scored best (88.7 %) and is the default engine. Your Keras model stays available via `MALWARE_ENGINE=keras`. |
| 2, 5 | `training/train_url_model.py` de-duplicates, drops conflicting labels, and reports metrics on held-out *domains*. |
| 3 | `detectors/url_features.py::normalize_url` strips scheme, credentials, `www.` and fragment and ensures a `/` after the host. Used in training and inference. |
| 4 | URL model retrained on the provided file; artifacts are `url_model.joblib` (vectorizer + classifier in one pipeline) and a model card. |
| 8-11 | `config.py`, absolute paths, models loaded once at startup, label mapping read from the classifier / model card, debug off by default, 20 MB / 200,000 row limits, extension + binary-content + UTF-8 checks, specific error messages, secure filenames, no uploaded file ever written to disk, result files auto-deleted after 30 minutes. |
| 10 | Strict CSV contract: columns matched by name (reordering is reported), missing/non-numeric/infinite cells are rejected with their row and column, nothing is imputed. |
| 12 | New dashboard that writes all dynamic text with `textContent`, plus a Content-Security-Policy. |
| 13 | Pinned, mutually tested requirements; `waitress` (Windows) and `gunicorn` (Linux). |

## 4. What I could not verify

* **How well either model works on data outside these two files.** All measurements are on the provided datasets (CSV: 100 apps; URL: one list). No external test set was provided.
* **The original training file for the old URL pickle** (see problem 4).
* **How the 27 features are extracted from a real app.** The project contains no extractor; the app consumes pre-extracted rows.
* **Windows execution.** Everything was run on Linux. The instructions and code avoid Linux-only features, but I could not run them on Windows.
* **The dashboard in a real browser.** The JavaScript passes a syntax check and the HTML/JS/API contract is covered by tests, but I could not open it in a browser here.
