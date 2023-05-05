


class Threshold:

    def __init__(self) -> None:
        pass

    def apply_threshold(self, dataset, threshold, greater, column_name='temperature'):
        """
        Apply threshold to dataset
        :param dataset: dataset to apply threshold
        :param threshold: threshold value
        :param greater: boolean value to indicate if threshold is greater or lower
        :param column_name: column name to apply threshold
        :return: filtered dataset
        """

        count_1, count_0 = 0, 0
        if greater:
            for row in dataset:
                if row[column_name] > threshold:
                    row[column_name] = 1
                    count_1 += 1
                else:
                    row[column_name] = 0
                    count_0 += 1
        else:
            for row in dataset:
                if row[column_name] < threshold:
                    row[column_name] = 1
                    count_1 += 1
                else:
                    row[column_name] = 0
                    count_0 += 1

        return dataset, [count_0, count_1]