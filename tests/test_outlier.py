import pytest
from BioML.utilities.outlier import OutlierDetection

@pytest.fixture
def outlier_detection():
    return OutlierDetection("test_features.xlsx", "test_outliers.csv", "minmax", 0.06, 10, 1.0)

def test_outlier_detection_init(outlier_detection):
    assert outlier_detection.scaler == "minmax"
    assert outlier_detection.contamination == 0.06
    assert outlier_detection.feature_file == "test_features.xlsx"
    assert outlier_detection.num_threads == 10
    assert outlier_detection.num_feature == 1.0

def test_outlier_detection_initalize_models(outlier_detection):
    classifiers = outlier_detection.initalize_models()
    assert len(classifiers) == 8
    assert "iforest" in classifiers
    assert "knn" in classifiers
    assert "bagging" in classifiers
    assert "hbos" in classifiers
    assert "abod" in classifiers
    assert "pca" in classifiers
    assert "ocsvm" in classifiers
    assert "ecod" in classifiers

def test_outlier_detection_outlier(outlier_detection):
    transformed_x = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    prediction = outlier_detection.outlier(transformed_x)
    assert len(prediction) == 8
    assert "iforest" in prediction
    assert "knn" in prediction
    assert "bagging" in prediction
    assert "hbos" in prediction
    assert "abod" in prediction
    assert "pca" in prediction
    assert "ocsvm" in prediction
    assert "ecod" in prediction

def test_outlier_detection_counting(outlier_detection):
    prediction = {
        "iforest": [0, 1, 0],
        "knn": [1, 0, 1],
        "bagging": [0, 0, 1],
        "hbos": [1, 1, 0],
        "abod": [0, 1, 1],
        "pca": [1, 0, 0],
        "ocsvm": [0, 1, 0],
        "ecod": [1, 0, 1]
    }
    index = ["feature1", "feature2", "feature3"]
    counted = outlier_detection.counting(prediction, index)
    assert len(counted) == 3
    assert counted["feature1"] == 4
    assert counted["feature2"] == 4
    assert counted["feature3"] == 4

def test_outlier_detection_read_features(outlier_detection):
    book = ["Sheet1", "Sheet2"]
    excel_data = outlier_detection._read_features(book)
    assert len(excel_data) == 2
    assert "Sheet1" in excel_data
    assert "Sheet2" in excel_data

def test_outlier_detection_run(outlier_detection):
    summed = outlier_detection.run()
    assert len(summed) == 3
    assert summed["feature1"] == 4
    assert summed["feature2"] == 4
    assert summed["feature3"] == 4
