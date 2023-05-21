import pandas as pd
from collections import Counter
from pathlib import Path


class Threshold:

    def __init__(self, csv_file, output_csv, sep=","):
        self.csv_file = pd.read_csv(csv_file, sep=sep, index_col=0)
        self.output_csv = Path(output_csv)
        self.output_csv.parent.mkdir(exist_ok=True, parents=True)

    def apply_threshold(self, threshold, greater=True, column_name='temperature'):
        """
        Apply threshold to dataset
        :param threshold: threshold value
        :param greater: boolean value to indicate if threshold is greater or lower
        :param column_name: column name to apply threshold
        :return: filtered dataset
        """
        dataset = self.csv_file[column_name].copy()
        dataset = pd.to_numeric(dataset, errors="coerce")
        dataset.dropna(inplace=True)
        data = dataset.copy()
        if greater:
            data.loc[dataset >= threshold] = 1
            data.loc[dataset < threshold] = 0
        else:
            data.loc[dataset <= threshold] = 1
            data.loc[dataset > threshold] = 0
        print(f"using the threshold {threshold} returns these proportions", Counter(data))
        return data

    def save_csv(self, threshold, greater=True, column_name='temperature'):
        data = self.apply_threshold(threshold, greater, column_name)
        data.to_csv(self.output_csv)