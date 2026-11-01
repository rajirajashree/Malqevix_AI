import os

import numpy as np
import pandas as pd
import pytest

from detectors.errors import FileValidationError, ModelLoadError
from detectors.file_detector import FileDetector, prepare_features, _safe_cell
from detectors.file_features import FEATURE_COLUMNS, PASSTHROUGH_COLUMNS
from conftest import ROOT, good_row


def frame(rows=None):
    return pd.DataFrame(rows if rows is not None else [good_row(), good_row(state=5)])


def test_feature_contract_is_27_unique_columns():
    assert len(FEATURE_COLUMNS) == 27 and len(set(FEATURE_COLUMNS)) == 27
    assert not set(FEATURE_COLUMNS) & set(PASSTHROUGH_COLUMNS)


def test_valid_frame_returns_float_array_in_training_order():
    arr, ignored, warnings = prepare_features(frame(), 100)
    assert arr.shape == (2, 27) and arr.dtype == np.float64 and ignored == [] and warnings == []
    assert arr[1, FEATURE_COLUMNS.index('state')] == 5


def test_reordered_columns_are_matched_by_name_and_reported():
    df = frame()[list(reversed(FEATURE_COLUMNS))]
    arr, _, warnings = prepare_features(df, 100)
    assert arr[1, FEATURE_COLUMNS.index('state')] == 5
    assert any('different order' in w for w in warnings)


def test_too_few_columns():
    with pytest.raises(FileValidationError, match='Missing 2 required column'):
        prepare_features(frame().drop(columns=['state', 'lock']), 100)


def test_too_many_columns():
    df = frame(); df['bonus'] = 1
    with pytest.raises(FileValidationError, match='Unexpected column.*bonus'):
        prepare_features(df, 100)


def test_headerless_file_gets_a_helpful_message():
    df = pd.DataFrame(np.zeros((3, 27)))
    with pytest.raises(FileValidationError, match='first row of the CSV must be a header'):
        prepare_features(df, 100)


def test_known_dataset_columns_are_kept_but_not_scored():
    df = frame(); df['hash'] = ['a', 'b']; df['classification'] = ['benign', 'malware']
    arr, ignored, warnings = prepare_features(df, 100)
    assert arr.shape[1] == 27 and set(ignored) == {'hash', 'classification'} and warnings


def test_non_numeric_values_are_located():
    df = frame(); df['state'] = df['state'].astype(object); df.loc[1, 'state'] = 'abc'
    with pytest.raises(FileValidationError, match=r'non-numeric.*row 3 / state'):
        prepare_features(df, 100)


def test_missing_values_are_rejected_not_imputed():
    df = frame(); df.loc[0, 'prio'] = np.nan
    with pytest.raises(FileValidationError, match=r'empty cell.*row 2 / prio'):
        prepare_features(df, 100)


def test_infinite_values_rejected():
    df = frame().astype(float); df.loc[0, 'prio'] = np.inf
    with pytest.raises(FileValidationError, match='infinite'):
        prepare_features(df, 100)


def test_header_only_and_row_limit():
    with pytest.raises(FileValidationError, match='no data rows'):
        prepare_features(pd.DataFrame(columns=FEATURE_COLUMNS), 100)
    with pytest.raises(FileValidationError, match='limit is 1'):
        prepare_features(frame(), 1)


def test_string_numbers_are_accepted():
    df = frame().astype(str)
    arr, _, _ = prepare_features(df, 100)
    assert arr[1, FEATURE_COLUMNS.index('state')] == 5


def test_prediction_shape_label_mapping_and_columns(toy_model_dir):
    det = FileDetector(toy_model_dir)
    out, report = det.predict_frame(frame([good_row(state=0.9), good_row(state=0.1), good_row(state=0.95)]), 100)
    assert list(out['prediction']) == ['Malware', 'Benign', 'Malware']  # class index 1 == Malware in the toy data
    assert out['malware_score'].between(0, 1).all() and len(out) == 3
    assert list(out.columns[-2:]) == ['malware_score', 'prediction']
    s = report['summary']
    assert (s['rows'], s['malware'], s['benign']) == (3, 2, 1) and sum(s['score_histogram']) == 3
    assert len(report['preview']) == 3 and report['preview'][0]['row'] == 2


def test_original_input_columns_and_order_preserved_in_output(toy_model_dir):
    df = frame(); df.insert(0, 'hash', ['x', 'y'])
    out, _ = FileDetector(toy_model_dir).predict_frame(df, 100)
    assert list(out.columns[:28]) == ['hash'] + FEATURE_COLUMNS


def test_formula_injection_is_neutralised():
    assert _safe_cell('=HYPERLINK("http://x")') == "'=HYPERLINK(\"http://x\")"
    assert _safe_cell('abc') == 'abc' and _safe_cell(5) == 5


def test_missing_files_raise_clear_error(tmp_path):
    with pytest.raises(ModelLoadError, match='malware_model_card.json'):
        FileDetector(str(tmp_path))


def test_missing_model_file_with_card_present(toy_model_dir, tmp_path):
    (tmp_path / 'malware_model_card.json').write_text(open(os.path.join(toy_model_dir, 'malware_model_card.json')).read())
    with pytest.raises(ModelLoadError, match='malware_rf.joblib'):
        FileDetector(str(tmp_path))


def test_card_with_different_feature_order_is_rejected(toy_model_dir, tmp_path):
    import json
    card = json.load(open(os.path.join(toy_model_dir, 'malware_model_card.json'))); card['features'] = card['features'][::-1]
    (tmp_path / 'malware_model_card.json').write_text(json.dumps(card))
    with pytest.raises(ModelLoadError, match='different feature set/order'):
        FileDetector(str(tmp_path))


def test_unknown_engine_rejected(toy_model_dir):
    with pytest.raises(ModelLoadError, match='Unknown MALWARE_ENGINE'):
        FileDetector(toy_model_dir, engine='magic')


def labelled_sample():
    df = pd.read_csv(os.path.join(ROOT, 'sample_data', 'labeled_sample_from_training_data.csv'))
    return df, (df['classification'] == 'malware').to_numpy()


@pytest.mark.integration
def test_real_sklearn_model_mapping_and_shape():
    df, truth = labelled_sample()
    det = FileDetector(os.path.join(ROOT, 'model'), 'sklearn')
    out, _ = det.predict_frame(df, 1000)
    assert len(out) == len(df) and set(out['prediction']) <= {'Malware', 'Benign'}
    # sample rows come from the TRAINING apps: this guards the label mapping only, it is not an accuracy measurement
    assert ((out['prediction'] == 'Malware').to_numpy() == truth).mean() > 0.9


@pytest.mark.integration
@pytest.mark.keras
def test_real_keras_model_mapping_probabilities_and_shape():
    pytest.importorskip('tensorflow')
    df, truth = labelled_sample()
    det = FileDetector(os.path.join(ROOT, 'model'), 'keras')
    arr, _, _ = prepare_features(df.copy(), 1000)
    scores = det.score(arr)
    assert scores.shape == (len(df),) and ((scores >= 0) & (scores <= 1)).all()
    assert ((scores >= 0.5) == truth).mean() > 0.9  # label-mapping guard on training apps, not an accuracy claim
