class NotSupportedDataError(Exception):
    """
    When the file provided is not supported yet
    """

class DifferentLabelFeatureIndexError(Exception):
    """
    When the label and feature index are different
    """