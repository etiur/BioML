import pandas as pd
from collections import Counter


class Threshold:

    def __init__(self, csv_file, sep=","):
        self.csv_file = pd.read_csv(csv_file, sep=sep, index_col=0)

    def apply_threshold(self, threshold, greater, column_name='temperature'):
        """
        Apply threshold to dataset
        :param dataset: dataset to apply threshold
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
        print(Counter(dataset))
        return data