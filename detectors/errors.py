class ModelLoadError(RuntimeError):
    """A model artifact is missing, unreadable or inconsistent with its model card."""


class InvalidURLError(ValueError):
    """The submitted URL string is not something we can analyse."""


class FileValidationError(ValueError):
    """The uploaded CSV does not satisfy the model's input contract."""
