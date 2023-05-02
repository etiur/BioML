from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
import pandas as pd
from pathlib import Path


def scale(scaler, X_train, X_test=None):
    """
    Scale the features using RobustScaler
    """
    scaler_dict = {"robust": RobustScaler(), "standard": StandardScaler(), "minmax": MinMaxScaler()}
    transformed = scaler_dict[scaler].fit_transform(X_train)
    transformed = pd.DataFrame(transformed, index=X_train.index, columns=X_train.columns)
    if not X_test:
        return transformed, scaler_dict
    else:
        test_x = scaler_dict[scaler].transform(X_test)
        test_x = pd.DataFrame(test_x, index=X_test.index, columns=X_test.columns)
        return transformed, scaler_dict, test_x


def analyse_composition(dataframe):
    col = dataframe.columns
    count = 0
    for x in col:
        if "pssm" in x or "tpc" in x or "eedp" in x or "edp" in x:
            count += 1
    print(f"POSSUM: {count}, iFeature: {len(col) - count}")
    return count, len(col) - count


def write_excel(file, dataframe, sheet_name, overwrite=False):
    if not str(file).endswith(".xlsx"):
        file = Path(file).with_suffix(".xlsx")
    if overwrite or not Path(file).exists():
        mode = "w"
    else:
        mode = "a"
    with pd.ExcelWriter(file, mode=mode, engine="openpyxl") as writer:
        dataframe.to_excel(writer, sheet_name=sheet_name)
