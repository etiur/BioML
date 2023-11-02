class NotSupportedDataError(Exception):
    """
    When the file provided is not supported yet
    """

class DifferentLabelFeatureIndexError(Exception):
    """
    When the label and feature index are different
    """

class SheetsNotFoundInExcelError(Exception):
    """
    When the sheets are not found in the excel file
    """