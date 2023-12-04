class DifferentLabelFeatureIndexError(Exception):
    """
    When the label and feature index are different
    """

class SheetsNotFoundInExcelError(Exception):
    """
    When the sheets are not found in the excel file
    """

class InsufficientSamplesError(Exception):
    """
    When the number of samples left for the test set is less than the defined number
    """

class InsufficientClustersError(Exception):
    """
    When the number of clusters is less than the number of folds
    """