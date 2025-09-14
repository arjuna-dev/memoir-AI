"""
Custom exceptions for MemoirAI library.
"""

from typing import Any


class MemoirAIError(Exception):
    """Base exception for all MemoirAI errors."""

    pass


class ConfigurationError(MemoirAIError):
    """Raised when configuration parameters are invalid."""

    def __init__(self, message: str, parameter: str = None, suggested_fix: str = None):
        self.parameter = parameter
        self.suggested_fix = suggested_fix

        full_message = f"Configuration Error: {message}"
        if parameter:
            full_message += f" (Parameter: {parameter})"
        if suggested_fix:
            full_message += f" Suggested fix: {suggested_fix}"

        super().__init__(full_message)


class ClassificationError(MemoirAIError):
    """Raised when LLM classification fails."""

    def __init__(self, message: str, chunk_id: int = None, retry_count: int = 0):
        self.chunk_id = chunk_id
        self.retry_count = retry_count

        full_message = f"Classification Error: {message}"
        if chunk_id is not None:
            full_message += f" (Chunk ID: {chunk_id})"
        if retry_count > 0:
            full_message += f" (Retries: {retry_count})"

        super().__init__(full_message)


class DatabaseError(MemoirAIError):
    """Raised when database operations fail."""

    def __init__(self, message: str, operation: str = None, table: str = None):
        self.operation = operation
        self.table = table

        full_message = f"Database Error: {message}"
        if operation:
            full_message += f" (Operation: {operation})"
        if table:
            full_message += f" (Table: {table})"

        super().__init__(full_message)


class TokenBudgetError(MemoirAIError):
    """Raised when token budget constraints cannot be met."""

    def __init__(
        self, message: str, required_tokens: int = None, available_tokens: int = None
    ):
        self.required_tokens = required_tokens
        self.available_tokens = available_tokens

        full_message = f"Token Budget Error: {message}"
        if required_tokens and available_tokens:
            full_message += (
                f" (Required: {required_tokens}, Available: {available_tokens})"
            )

        super().__init__(full_message)


class ValidationError(MemoirAIError):
    """Raised when input validation fails."""

    def __init__(self, message: str, field: str = None, value: Any = None):
        self.field = field
        self.value = value

        full_message = f"Validation Error: {message}"
        if field:
            full_message += f" (Field: {field})"
        if value is not None:
            full_message += f" (Value: {value})"

        super().__init__(full_message)


class LLMError(MemoirAIError):
    """Raised when LLM operations fail."""

    def __init__(
        self,
        message: str,
        model: str = None,
        error_type: str = None,
        retry_suggested: bool = True,
    ):
        self.model = model
        self.error_type = error_type
        self.retry_suggested = retry_suggested

        full_message = f"LLM Error: {message}"
        if model:
            full_message += f" (Model: {model})"
        if error_type:
            full_message += f" (Type: {error_type})"

        super().__init__(full_message)
