class ModelNotFoundError(Exception):
    """Raised when a provided model not in the configured available models (config/models.toml)."""
    pass