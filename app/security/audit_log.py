import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import uuid
from enum import Enum

from app.utils.logger import get_logger
from app.db.postgres import PostgreSQLManager

logger = get_logger(__name__)


class AuditAction(str, Enum):
    """Audit actions."""
    # User actions
    LOGIN = "login"
    LOGOUT = "logout"
    TOKEN_CREATE = "token_create"
    TOKEN_REVOKE = "token_revoke"

    # Transaction actions
    TRANSACTION_VIEW = "transaction_view"
    TRANSACTION_CREATE = "transaction_create"
    TRANSACTION_UPDATE = "transaction_update"
    TRANSACTION_DELETE = "transaction_delete"

    # Fraud case actions
    FRAUD_CASE_VIEW = "fraud_case_view"
    FRAUD_CASE_CREATE = "fraud_case_create"
    FRAUD_CASE_UPDATE = "fraud_case_update"
    FRAUD_CASE_RESOLVE = "fraud_case_resolve"

    # Policy actions
    POLICY_SEARCH = "policy_search"
    POLICY_VIEW = "policy_view"
    POLICY_CREATE = "policy_create"
    POLICY_UPDATE = "policy_update"
    POLICY_DELETE = "policy_delete"

    # System actions
    CONFIG_UPDATE = "config_update"
    USER_CREATE = "user_create"
    USER_UPDATE = "user_update"
    USER_DELETE = "user_delete"

    # Investigation actions
    INVESTIGATION_START = "investigation_start"
    INVESTIGATION_VIEW = "investigation_view"
    INVESTIGATION_UPDATE = "investigation_update"


class AuditSeverity(str, Enum):
    """Audit log severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AuditLogger:
    """Production-grade audit logging system."""

    def __init__(self):
        self.db = PostgreSQLManager()

        # Audit configuration
        self.config = {
            "enabled": True,
            "log_level": AuditSeverity.INFO,
            "mask_pii": True,
            "retention_days": 365
        }

        # Action severity mapping
        self.action_severity = {
            AuditAction.LOGIN: AuditSeverity.INFO,
            AuditAction.LOGOUT: AuditSeverity.INFO,
            AuditAction.TOKEN_CREATE: AuditSeverity.INFO,
            AuditAction.TOKEN_REVOKE: AuditSeverity.WARNING,

            AuditAction.TRANSACTION_VIEW: AuditSeverity.INFO,
            AuditAction.TRANSACTION_CREATE: AuditSeverity.INFO,
            AuditAction.TRANSACTION_UPDATE: AuditSeverity.WARNING,
            AuditAction.TRANSACTION_DELETE: AuditSeverity.ERROR,

            AuditAction.FRAUD_CASE_CREATE: AuditSeverity.WARNING,
            AuditAction.FRAUD_CASE_RESOLVE: AuditSeverity.INFO,

            AuditAction.POLICY_CREATE: AuditSeverity.WARNING,
            AuditAction.POLICY_DELETE: AuditSeverity.ERROR,

            AuditAction.USER_CREATE: AuditSeverity.WARNING,
            AuditAction.USER_DELETE: AuditSeverity.CRITICAL,

            AuditAction.INVESTIGATION_START: AuditSeverity.INFO
        }

    async def initialize(self):
        """Initialize audit logger."""
        await self.db.initialize()
        logger.info("Audit logger initialized")

    async def log(self,
                  user_id: str,
                  action: AuditAction,
                  resource_type: str,
                  resource_id: str,
                  request_body: Optional[Dict[str, Any]] = None,
                  response_body: Optional[Dict[str, Any]] = None,
                  status_code: Optional[int] = None,
                  ip_address: Optional[str] = None,
                  user_agent: Optional[str] = None,
                  additional_info: Optional[Dict[str, Any]] = None,
                  severity: Optional[AuditSeverity] = None):
        """Log an audit event."""
        if not self.config["enabled"]:
            return

        # Determine severity
        if not severity:
            severity = self.action_severity.get(action, AuditSeverity.INFO)

        # Check if we should log based on severity
        if not self._should_log(severity):
            return

        try:
            # Prepare audit data
            audit_data = {
                "user_id": user_id,
                "action": action,
                "resource_type": resource_type,
                "resource_id": resource_id,
                "request_body": self._sanitize_data(request_body) if request_body else None,
                "response_body": self._sanitize_data(response_body) if response_body else None,
                "status_code": status_code,
                "ip_address": ip_address,
                "user_agent": user_agent,
                "severity": severity.value,
                "timestamp": datetime.now().isoformat(),
                "additional_info": additional_info or {}
            }

            # Log to database
            await self.db.log_audit_event(audit_data)

            # Also log to application log
            log_message = f"Audit: {action} by {user_id} on {resource_type}/{resource_id}"
            if severity == AuditSeverity.ERROR or severity == AuditSeverity.CRITICAL:
                logger.error(log_message)
            elif severity == AuditSeverity.WARNING:
                logger.warning(log_message)
            else:
                logger.info(log_message)

        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")

    def _should_log(self, severity: AuditSeverity) -> bool:
        """Check if we should log based on severity level."""
        severity_levels = {
            AuditSeverity.INFO: 1,
            AuditSeverity.WARNING: 2,
            AuditSeverity.ERROR: 3,
            AuditSeverity.CRITICAL: 4
        }

        config_level = severity_levels.get(self.config["log_level"], 1)
        event_level = severity_levels.get(severity, 1)

        return event_level >= config_level

    def _sanitize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize data for logging (mask PII, remove sensitive info)."""
        if not data:
            return {}

        sanitized = data.copy()

        # Fields to always mask/remove
        sensitive_fields = {
            'password', 'token', 'secret', 'api_key', 'private_key',
            'credit_card', 'cvv', 'ssn', 'dob', 'address', 'phone'
        }

        def sanitize_dict(d: Dict[str, Any]) -> Dict[str, Any]:
            result = {}
            for key, value in d.items():
                key_lower = key.lower()

                # Check if field is sensitive
                is_sensitive = any(sensitive in key_lower for sensitive in sensitive_fields)

                if is_sensitive:
                    result[key] = "[REDACTED]"
                elif isinstance(value, dict):
                    result[key] = sanitize_dict(value)
                elif isinstance(value, list):
                    result[key] = [
                        sanitize_dict(item) if isinstance(item, dict)
                        else "[REDACTED]" if is_sensitive
                        else item
                        for item in value
                    ]
                else:
                    result[key] = value

            return result

        return sanitize_dict(sanitized)

    async def query_logs(self,
                         start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None,
                         user_id: Optional[str] = None,
                         action: Optional[str] = None,
                         resource_type: Optional[str] = None,
                         resource_id: Optional[str] = None,
                         severity: Optional[str] = None,
                         limit: int = 100) -> List[Dict[str, Any]]:
        """Query audit logs with filters."""
        try:
            # Build query
            query = "SELECT * FROM audit_logs WHERE 1=1"
            params = []

            if start_time:
                query += " AND created_at >= %s"
                params.append(start_time)

            if end_time:
                query += " AND created_at <= %s"
                params.append(end_time)

            if user_id:
                query += " AND user_id = %s"
                params.append(user_id)

            if action:
                query += " AND action = %s"
                params.append(action)

            if resource_type:
                query += " AND resource_type = %s"
                params.append(resource_type)

            if resource_id:
                query += " AND resource_id = %s"
                params.append(resource_id)

            if severity:
                query += " AND severity = %s"
                params.append(severity)

            query += " ORDER BY created_at DESC LIMIT %s"
            params.append(limit)

            results = await self.db.execute_custom_query(query, tuple(params))
            return results

        except Exception as e:
            logger.error(f"Failed to query audit logs: {e}")
            return []

    async def get_statistics(self,
                             start_time: Optional[datetime] = None,
                             end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """Get audit log statistics."""
        try:
            query = """
                SELECT 
                    COUNT(*) as total_logs,
                    COUNT(DISTINCT user_id) as unique_users,
                    COUNT(CASE WHEN severity = 'critical' THEN 1 END) as critical_count,
                    COUNT(CASE WHEN severity = 'error' THEN 1 END) as error_count,
                    COUNT(CASE WHEN severity = 'warning' THEN 1 END) as warning_count,
                    COUNT(CASE WHEN severity = 'info' THEN 1 END) as info_count,
                    MIN(created_at) as earliest_log,
                    MAX(created_at) as latest_log
                FROM audit_logs
                WHERE 1=1
            """

            params = []

            if start_time:
                query += " AND created_at >= %s"
                params.append(start_time)

            if end_time:
                query += " AND created_at <= %s"
                params.append(end_time)

            results = await self.db.execute_custom_query(query, tuple(params))

            if results:
                stats = results[0]

                # Add action frequency
                action_query = """
                    SELECT action, COUNT(*) as count
                    FROM audit_logs
                    WHERE 1=1
                """

                if start_time:
                    action_query += " AND created_at >= %s"
                if end_time:
                    action_query += " AND created_at <= %s"

                action_query += " GROUP BY action ORDER BY count DESC LIMIT 10"

                action_results = await self.db.execute_custom_query(action_query, tuple(params))

                return {
                    "period": {
                        "start": start_time,
                        "end": end_time
                    },
                    "summary": stats,
                    "top_actions": action_results
                }

            return {}

        except Exception as e:
            logger.error(f"Failed to get audit statistics: {e}")
            return {}

    async def export_logs(self,
                          start_time: datetime,
                          end_time: datetime,
                          format: str = "json") -> str:
        """Export audit logs in specified format."""
        try:
            logs = await self.query_logs(
                start_time=start_time,
                end_time=end_time,
                limit=10000  # Limit for export
            )

            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "period": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat()
                },
                "log_count": len(logs),
                "logs": logs
            }

            if format.lower() == "json":
                return json.dumps(export_data, indent=2, default=str)
            elif format.lower() == "csv":
                # Convert to CSV
                import csv
                import io

                output = io.StringIO()
                writer = csv.DictWriter(output, fieldnames=[
                    "timestamp", "user_id", "action", "resource_type",
                    "resource_id", "severity", "status_code"
                ])

                writer.writeheader()
                for log in logs:
                    writer.writerow({
                        "timestamp": log.get("created_at"),
                        "user_id": log.get("user_id"),
                        "action": log.get("action"),
                        "resource_type": log.get("resource_type"),
                        "resource_id": log.get("resource_id"),
                        "severity": log.get("severity"),
                        "status_code": log.get("status_code")
                    })

                return output.getvalue()
            else:
                raise ValueError(f"Unsupported format: {format}")

        except Exception as e:
            logger.error(f"Failed to export audit logs: {e}")
            return ""

    async def cleanup_old_logs(self, days: Optional[int] = None):
        """Clean up old audit logs."""
        try:
            retention_days = days or self.config["retention_days"]
            cutoff_date = datetime.now() - timedelta(days=retention_days)

            query = "DELETE FROM audit_logs WHERE created_at < %s"
            await self.db.execute_custom_query(query, (cutoff_date,))

            logger.info(f"Cleaned up audit logs older than {retention_days} days")

        except Exception as e:
            logger.error(f"Failed to cleanup old audit logs: {e}")

    async def log_security_event(self,
                                 event_type: str,
                                 description: str,
                                 user_id: Optional[str] = None,
                                 severity: AuditSeverity = AuditSeverity.WARNING,
                                 details: Optional[Dict] = None):
        """Log security-specific event."""
        await self.log(
            user_id=user_id or "system",
            action=AuditAction(event_type),
            resource_type="security",
            resource_id=str(uuid.uuid4()),
            request_body=details,
            severity=severity,
            additional_info={"security_event": True, "description": description}
        )

    async def log_investigation(self,
                                investigation_id: str,
                                user_id: str,
                                action: str,
                                details: Dict[str, Any]):
        """Log investigation-related event."""
        await self.log(
            user_id=user_id,
            action=AuditAction(f"investigation_{action}"),
            resource_type="investigation",
            resource_id=investigation_id,
            request_body=details,
            severity=AuditSeverity.INFO,
            additional_info={"investigation": True}
        )


# Global instance
audit_logger = AuditLogger()


async def init_audit_logger():
    """Initialize audit logger."""
    await audit_logger.initialize()