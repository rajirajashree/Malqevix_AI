"""CSV malware detector. Input: rows of 27 runtime/process features (see file_features.py)."""
import json
import logging
import os

import joblib
import numpy as np
import pandas as pd

from .errors import FileValidationError, ModelLoadError
from .file_features import FEATURE_COLUMNS, PASSTHROUGH_COLUMNS

log = logging.getLogger(__name__)
LABELS = {0: 'Benign', 1: 'Malware'}


def _example_locations(mask, columns, limit=5):
    rows, cols = np.where(mask)
    return ', '.join(f'row {int(r) + 2} / {columns[c]}' for r, c in list(zip(rows, cols))[:limit])  # +2: header is row 1


def prepare_features(df, max_rows):
    """Validate an uploaded frame against the model contract.

    Returns (features as float64 ndarray in training order, list of ignored columns, list of warnings).
    Raises FileValidationError with an actionable message. Never invents or imputes values.
    """
    df.columns = [str(c).strip() for c in df.columns]
    if df.shape[0] == 0:
        raise FileValidationError('The file has a header but no data rows.')
    if df.shape[0] > max_rows:
        raise FileValidationError(f'The file has {df.shape[0]:,} rows; the limit is {max_rows:,}.')

    cols = list(df.columns)
    missing = [c for c in FEATURE_COLUMNS if c not in cols]
    unexpected = [c for c in cols if c not in FEATURE_COLUMNS and c not in PASSTHROUGH_COLUMNS]
    if missing or unexpected:
        if len(missing) == len(FEATURE_COLUMNS):
            raise FileValidationError(
                f'No column names match the model. The first row of the CSV must be a header with these {len(FEATURE_COLUMNS)} names: '
                + ', '.join(FEATURE_COLUMNS) + f'. Found {df.shape[1]} columns.')
        parts = []
        if missing:
            parts.append(f'Missing {len(missing)} required column(s): {", ".join(missing)}.')
        if unexpected:
            parts.append(f'Unexpected column(s): {", ".join(unexpected)}.')
        raise FileValidationError(' '.join(parts) + f' The model needs exactly the {len(FEATURE_COLUMNS)} features it was trained on.')

    warnings = []
    ignored = [c for c in cols if c in PASSTHROUGH_COLUMNS]
    if ignored:
        warnings.append('Columns kept in the output but not used by the model: ' + ', '.join(ignored) + '.')
    if [c for c in cols if c in FEATURE_COLUMNS] != FEATURE_COLUMNS:
        warnings.append('The feature columns were in a different order; they were matched by name to the training order.')

    feats = df[FEATURE_COLUMNS]
    numeric = feats.apply(pd.to_numeric, errors='coerce')
    empty = feats.isna().to_numpy()
    non_numeric = numeric.isna().to_numpy() & ~empty
    if empty.any():
        raise FileValidationError(f'{int(empty.sum())} empty cell(s) found (e.g. {_example_locations(empty, FEATURE_COLUMNS)}). '
                                  'Missing values are not filled in; fix or remove those rows.')
    if non_numeric.any():
        raise FileValidationError(f'{int(non_numeric.sum())} non-numeric cell(s) found (e.g. {_example_locations(non_numeric, FEATURE_COLUMNS)}). '
                                  'All 27 features must be numbers.')
    arr = numeric.to_numpy(dtype='float64')
    if not np.isfinite(arr).all():
        raise FileValidationError('The file contains infinite values (inf/-inf). Remove or correct them.')
    return arr, ignored, warnings


def _safe_cell(v):
    """Neutralise spreadsheet formula injection in text cells that are echoed back in the result CSV."""
    if isinstance(v, str) and v[:1] in ('=', '+', '-', '@', '\t', '\r'):
        return "'" + v
    return v


class FileDetector:
    def __init__(self, model_dir, engine='sklearn', threshold=0.5):
        self.engine = engine
        self.threshold = float(threshold)
        card_path = os.path.join(model_dir, 'malware_model_card.json')
        self._require(card_path, model_dir)
        with open(card_path, encoding='utf-8') as fh:
            self.card = json.load(fh)
        if self.card.get('features') != FEATURE_COLUMNS:
            raise ModelLoadError('malware_model_card.json lists a different feature set/order than detectors/file_features.py.')
        if engine == 'sklearn':
            self._load_sklearn(model_dir)
        elif engine == 'keras':
            self._load_keras(model_dir)
        else:
            raise ModelLoadError(f'Unknown MALWARE_ENGINE "{engine}". Use "sklearn" or "keras".')

    @staticmethod
    def _require(path, model_dir):
        if not os.path.isfile(path):
            raise ModelLoadError(f'Missing malware model file: {os.path.basename(path)} (expected in {model_dir}). '
                                 'Run training/train_malware_model.py or copy the file into the model folder.')

    def _load_sklearn(self, model_dir):
        path = os.path.join(model_dir, 'malware_rf.joblib')
        self._require(path, model_dir)
        try:
            self.model = joblib.load(path)
        except Exception as exc:
            raise ModelLoadError(f'Could not load malware_rf.joblib: {exc}. Install the versions in requirements.txt or retrain.') from exc
        if getattr(self.model, 'n_features_in_', None) != len(FEATURE_COLUMNS):
            raise ModelLoadError('malware_rf.joblib does not expect 27 features.')
        classes = list(self.model.classes_)
        if classes != [0, 1]:
            raise ModelLoadError(f'Unexpected class labels {classes}; expected [0, 1] (0 = Benign, 1 = Malware).')
        self._malware_col = classes.index(1)
        self.algorithm = 'Random Forest (scikit-learn)'

    def _load_keras(self, model_dir):
        h5, scaler_path = (os.path.join(model_dir, n) for n in ('binary_malware_detector_model.h5', 'scaler.pkl'))
        self._require(h5, model_dir)
        self._require(scaler_path, model_dir)
        try:
            import tensorflow as tf  # imported lazily so the default engine does not need TensorFlow
            self.model = tf.keras.models.load_model(h5, compile=False)
            self.scaler = joblib.load(scaler_path)
        except Exception as exc:
            raise ModelLoadError(f'Could not load the Keras model/scaler: {exc}. Check that tensorflow and scikit-learn match requirements.txt.') from exc
        if self.model.input_shape[-1] != len(FEATURE_COLUMNS) or self.model.output_shape[-1] != 2:
            raise ModelLoadError(f'Keras model shape {self.model.input_shape} -> {self.model.output_shape} is not 27 -> 2.')
        names = getattr(self.scaler, 'feature_names_in_', None)
        if self.scaler.n_features_in_ != len(FEATURE_COLUMNS) or (names is not None and list(names) != FEATURE_COLUMNS):
            raise ModelLoadError('scaler.pkl was not fitted on the same 27 features in the same order.')
        self._malware_col = 1
        self.algorithm = 'Neural network (Keras, original model)'

    def score(self, features):
        """Malware probability for each row of an (n, 27) float array."""
        if features.ndim != 2 or features.shape[1] != len(FEATURE_COLUMNS):
            raise FileValidationError(f'Expected an array with {len(FEATURE_COLUMNS)} columns, got shape {features.shape}.')
        if self.engine == 'sklearn':
            return self.model.predict_proba(pd.DataFrame(features, columns=FEATURE_COLUMNS))[:, self._malware_col]
        scaled = self.scaler.transform(pd.DataFrame(features, columns=FEATURE_COLUMNS))
        out = np.asarray(self.model.predict(scaled, verbose=0, batch_size=4096))
        if out.shape != (features.shape[0], 2) or not np.allclose(out.sum(axis=1), 1.0, atol=1e-3):
            raise RuntimeError('The Keras model did not return class probabilities (rows do not sum to 1).')
        return out[:, self._malware_col]

    def predict_frame(self, df, max_rows, preview_rows=100):
        features, ignored, warnings = prepare_features(df, max_rows)
        scores = self.score(features)
        is_malware = scores >= self.threshold
        out = df.copy()
        for c in out.columns[out.dtypes == object]:
            out[c] = out[c].map(_safe_cell)
        out['malware_score'] = np.round(scores, 4)
        out['prediction'] = np.where(is_malware, LABELS[1], LABELS[0])
        n, mal = len(out), int(is_malware.sum())
        hist, _ = np.histogram(scores, bins=10, range=(0, 1))
        mean = float(scores.mean())
        summary = {
            'rows': n, 'malware': mal, 'benign': n - mal, 'malware_pct': round(100 * mal / n, 2),
            'mean_score': round(mean, 4), 'threshold': self.threshold, 'score_histogram': [int(h) for h in hist],
            'file_level_note': ('Average score across all rows. Each row is one 1 ms snapshot of a process, so this is only '
                                'meaningful if every row comes from the same app.'),
            'file_level_label': LABELS[int(mean >= self.threshold)],
        }
        preview = [{'row': int(i) + 2, 'malware_score': float(s), 'prediction': p}
                   for i, (s, p) in enumerate(zip(out['malware_score'].head(preview_rows), out['prediction'].head(preview_rows)))]
        return out, {'summary': summary, 'preview': preview, 'warnings': warnings, 'ignored_columns': ignored}

    @property
    def info(self):
        c = self.card
        return {'kind': 'malware', 'engine': self.engine, 'algorithm': self.algorithm, 'created': c.get('created'),
                'threshold': self.threshold, 'features': FEATURE_COLUMNS, 'grouped_cv_metrics': c.get('grouped_cv_metrics')}
