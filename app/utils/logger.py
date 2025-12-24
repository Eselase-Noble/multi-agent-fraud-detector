import logging
import sys
import json
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        if hasattr(record, "extra"):
            log_data.update(record.extra)

        return json.dumps(log_data)


class ColorFormatter(logging.Formatter):
    """Color formatter for console output."""

    COLORS = {
        'DEBUG': '\033[36m',  # Cyan
        'INFO': '\033[32m',  # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',  # Red
        'CRITICAL': '\033[41m'  # Red background
    }
    RESET = '\033[0m'

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        message = super().format(record)
        return f"{color}{message}{self.RESET}"


def setup_logger(
        name: str = "intellifraud",
        level: str = "INFO",
        log_file: Optional[str] = None,
        json_format: bool = False
) -> logging.Logger:
    """
    Setup logger with console and optional file handlers.

    Args:
        name: Logger name
        level: Logging level
        log_file: Optional file path for file logging
        json_format: Use JSON format for structured logging

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Clear existing handlers
    logger.handlers.clear()

    # Set level
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)

    if json_format:
        console_formatter = JSONFormatter()
    else:
        console_formatter = ColorFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Create file handler if log_file specified
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get logger instance by name."""
    return logging.getLogger(name)


class AuditLoggerAdapter:
    """Adapter for audit logging."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def log_audit_event(self,
                        user_id: str,
                        action: str,
                        resource_type: str,
                        resource_id: str,
                        severity: str = "INFO",
                        details: Optional[Dict[str, Any]] = None):
        """Log audit event."""
        extra = {
            "audit": True,
            "user_id": user_id,
            "action": action,
            "resource_type": resource_type,
            "resource_id": resource_id,
            "severity": severity
        }

        if details:
            extra["details"] = details

        log_method = getattr(self.logger, severity.lower(), self.logger.info)
        log_method(f"Audit: {action} by {user_id} on {resource_type}/{resource_id}",
                   extra=extra)


class PerformanceLogger:
    """Performance monitoring logger."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.metrics = {}

    def start_timer(self, operation: str) -> str:
        """Start timer for operation."""
        timer_id = f"{operation}_{datetime.now().timestamp()}"
        self.metrics[timer_id] = {
            "operation": operation,
            "start_time": datetime.now(),
            "end_time": None,
            "duration": None
        }
        return timer_id

    def end_timer(self, timer_id: str, metadata: Optional[Dict] = None) -> float:
        """End timer and log duration."""
        if timer_id not in self.metrics:
            self.logger.warning(f"Timer {timer_id} not found")
            return 0.0

        metric = self.metrics[timer_id]
        metric["end_time"] = datetime.now()
        metric["duration"] = (metric["end_time"] - metric["start_time"]).total_seconds()

        if metadata:
            metric.update(metadata)

        # Log performance metric
        self.logger.info(
            f"Performance: {metric['operation']} took {metric['duration']:.3f}s",
            extra={"performance": metric}
        )

        return metric["duration"]

    def log_metric(self,
                   metric_name: str,
                   value: float,
                   metadata: Optional[Dict] = None):
        """Log a performance metric."""
        metric_data = {
            "metric": metric_name,
            "value": value,
            "timestamp": datetime.now().isoformat()
        }

        if metadata:
            metric_data.update(metadata)

        self.logger.info(
            f"Metric: {metric_name} = {value}",
            extra={"metric": metric_data}
        )


# Default logger setup
default_logger = setup_logger()


def get_default_logger() -> logging.Logger:
    """Get default logger instance."""
    return default_logger


# Convenience functions
def log_error(error: Exception, context: Optional[str] = None):
    """Log error with context."""
    logger = get_default_logger()
    message = f"Error: {str(error)}"
    if context:
        message = f"{context} - {message}"
    logger.error(message, exc_info=True)


def log_info(message: str, extra: Optional[Dict] = None):
    """Log info message with extra data."""
    logger = get_default_logger()
    if extra:
        logger.info(message, extra=extra)
    else:
        logger.info(message)


def log_warning(message: str, extra: Optional[Dict] = None):
    """Log warning message with extra data."""
    logger = get_default_logger()
    if extra:
        logger.warning(message, extra=extra)
    else:
        logger.warning(message)


def log_debug(message: str, extra: Optional[Dict] = None):
    """Log debug message with extra data."""
    logger = get_default_logger()
    if extra:
        logger.debug(message, extra=extra)
    else:
        logger.debug(message)