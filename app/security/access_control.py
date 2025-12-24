import uuid
from typing import List, Dict, Any, Optional, Set
from enum import Enum
from datetime import datetime, timedelta
import jwt
import bcrypt

from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class Role(str, Enum):
    """User roles with permissions."""
    ANALYST = "analyst"
    SUPERVISOR = "supervisor"
    ADMIN = "admin"
    AUDITOR = "auditor"
    SYSTEM = "system"


class Permission(str, Enum):
    """Individual permissions."""
    # Transaction permissions
    VIEW_TRANSACTIONS = "view_transactions"
    CREATE_TRANSACTIONS = "create_transactions"
    UPDATE_TRANSACTIONS = "update_transactions"
    DELETE_TRANSACTIONS = "delete_transactions"

    # Fraud case permissions
    VIEW_FRAUD_CASES = "view_fraud_cases"
    CREATE_FRAUD_CASES = "create_fraud_cases"
    UPDATE_FRAUD_CASES = "update_fraud_cases"
    RESOLVE_FRAUD_CASES = "resolve_fraud_cases"

    # Policy permissions
    VIEW_POLICIES = "view_policies"
    CREATE_POLICIES = "create_policies"
    UPDATE_POLICIES = "update_policies"
    DELETE_POLICIES = "delete_policies"

    # User management
    VIEW_USERS = "view_users"
    CREATE_USERS = "create_users"
    UPDATE_USERS = "update_users"
    DELETE_USERS = "delete_users"

    # System permissions
    VIEW_AUDIT_LOGS = "view_audit_logs"
    VIEW_SYSTEM_METRICS = "view_system_metrics"
    MANAGE_SYSTEM = "manage_system"


class AccessControl:
    """Production-grade access control system."""

    def __init__(self):
        self.jwt_secret = settings.JWT_SECRET if hasattr(settings,
                                                         'JWT_SECRET') else "default-secret-change-in-production"
        self.jwt_algorithm = "HS256"
        self.token_expiry = timedelta(hours=8)

        # Role-permission mapping
        self.role_permissions = {
            Role.ANALYST: {
                Permission.VIEW_TRANSACTIONS,
                Permission.CREATE_FRAUD_CASES,
                Permission.VIEW_FRAUD_CASES,
                Permission.VIEW_POLICIES
            },
            Role.SUPERVISOR: {
                Permission.VIEW_TRANSACTIONS,
                Permission.UPDATE_TRANSACTIONS,
                Permission.VIEW_FRAUD_CASES,
                Permission.CREATE_FRAUD_CASES,
                Permission.UPDATE_FRAUD_CASES,
                Permission.RESOLVE_FRAUD_CASES,
                Permission.VIEW_POLICIES,
                Permission.VIEW_USERS
            },
            Role.ADMIN: {
                Permission.VIEW_TRANSACTIONS,
                Permission.CREATE_TRANSACTIONS,
                Permission.UPDATE_TRANSACTIONS,
                Permission.DELETE_TRANSACTIONS,
                Permission.VIEW_FRAUD_CASES,
                Permission.CREATE_FRAUD_CASES,
                Permission.UPDATE_FRAUD_CASES,
                Permission.RESOLVE_FRAUD_CASES,
                Permission.VIEW_POLICIES,
                Permission.CREATE_POLICIES,
                Permission.UPDATE_POLICIES,
                Permission.DELETE_POLICIES,
                Permission.VIEW_USERS,
                Permission.CREATE_USERS,
                Permission.UPDATE_USERS,
                Permission.DELETE_USERS,
                Permission.VIEW_AUDIT_LOGS,
                Permission.VIEW_SYSTEM_METRICS,
                Permission.MANAGE_SYSTEM
            },
            Role.AUDITOR: {
                Permission.VIEW_TRANSACTIONS,
                Permission.VIEW_FRAUD_CASES,
                Permission.VIEW_POLICIES,
                Permission.VIEW_AUDIT_LOGS,
                Permission.VIEW_SYSTEM_METRICS
            },
            Role.SYSTEM: {
                Permission.VIEW_TRANSACTIONS,
                Permission.CREATE_TRANSACTIONS,
                Permission.UPDATE_TRANSACTIONS,
                Permission.VIEW_FRAUD_CASES,
                Permission.CREATE_FRAUD_CASES,
                Permission.VIEW_POLICIES,
                Permission.VIEW_SYSTEM_METRICS
            }
        }

        # User session store (in production, use Redis)
        self.sessions = {}

    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt."""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode(), salt)
        return hashed.decode()

    def verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify password against hash."""
        return bcrypt.checkpw(password.encode(), hashed_password.encode())

    def create_token(self, user_id: str, roles: List[Role], additional_claims: Optional[Dict] = None) -> str:
        """Create JWT token for user."""
        payload = {
            "sub": user_id,
            "roles": [role.value for role in roles],
            "exp": datetime.utcnow() + self.token_expiry,
            "iat": datetime.utcnow(),
            "jti": str(uuid.uuid4())  # Unique token ID
        }

        if additional_claims:
            payload.update(additional_claims)

        token = jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)

        # Store session
        self.sessions[payload["jti"]] = {
            "user_id": user_id,
            "roles": roles,
            "created_at": datetime.utcnow(),
            "expires_at": payload["exp"]
        }

        logger.info(f"Created token for user {user_id} with roles {roles}")
        return token

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token and return payload."""
        try:
            payload = jwt.decode(
                token,
                self.jwt_secret,
                algorithms=[self.jwt_algorithm]
            )

            # Check if token is in session store
            jti = payload.get("jti")
            if jti not in self.sessions:
                logger.warning(f"Token not found in session store: {jti}")
                return None

            # Check if session expired
            session = self.sessions[jti]
            if datetime.utcnow() > session["expires_at"]:
                logger.warning(f"Token expired: {jti}")
                del self.sessions[jti]
                return None

            return payload

        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None

    def has_permission(self, user_roles: List[Role], permission: Permission) -> bool:
        """Check if user has specific permission."""
        for role in user_roles:
            if role in self.role_permissions:
                if permission in self.role_permissions[role]:
                    return True
        return False

    def check_access(self,
                     token: str,
                     required_permission: Permission,
                     resource_id: Optional[str] = None) -> bool:
        """Check if user has access to resource."""
        payload = self.verify_token(token)
        if not payload:
            return False

        user_roles = [Role(role) for role in payload["roles"]]

        # Check permission
        if not self.has_permission(user_roles, required_permission):
            logger.warning(f"User {payload['sub']} lacks permission: {required_permission}")
            return False

        # Additional resource-based checks could go here
        if resource_id:
            # Check if user has access to specific resource
            pass

        return True

    def get_user_permissions(self, user_roles: List[Role]) -> Set[Permission]:
        """Get all permissions for user roles."""
        permissions = set()
        for role in user_roles:
            if role in self.role_permissions:
                permissions.update(self.role_permissions[role])
        return permissions

    def create_user(self,
                    username: str,
                    password: str,
                    roles: List[Role],
                    user_data: Optional[Dict] = None) -> Dict[str, Any]:
        """Create new user account."""
        # In production, this would save to database
        user_id = str(uuid.uuid4())

        user = {
            "user_id": user_id,
            "username": username,
            "password_hash": self.hash_password(password),
            "roles": [role.value for role in roles],
            "created_at": datetime.utcnow().isoformat(),
            "is_active": True,
            "metadata": user_data or {}
        }

        logger.info(f"Created user {username} with roles {roles}")
        return user

    def authenticate_user(self, username: str, password: str, user_store: Dict) -> Optional[str]:
        """Authenticate user and return token."""
        # Find user (in production, query database)
        user = None
        for stored_user in user_store.values():
            if stored_user["username"] == username:
                user = stored_user
                break

        if not user:
            logger.warning(f"Authentication failed: user {username} not found")
            return None

        if not user["is_active"]:
            logger.warning(f"Authentication failed: user {username} inactive")
            return None

        # Verify password
        if not self.verify_password(password, user["password_hash"]):
            logger.warning(f"Authentication failed: invalid password for {username}")
            return None

        # Create token
        roles = [Role(role) for role in user["roles"]]
        token = self.create_token(user["user_id"], roles)

        logger.info(f"User {username} authenticated successfully")
        return token

    def revoke_token(self, token: str) -> bool:
        """Revoke/blacklist a token."""
        try:
            payload = jwt.decode(
                token,
                self.jwt_secret,
                algorithms=[self.jwt_algorithm],
                options={"verify_exp": False}
            )

            jti = payload.get("jti")
            if jti in self.sessions:
                del self.sessions[jti]
                logger.info(f"Token revoked: {jti}")
                return True

            return False

        except jwt.InvalidTokenError:
            return False

    def revoke_all_user_tokens(self, user_id: str) -> int:
        """Revoke all tokens for a user."""
        revoked = 0
        tokens_to_remove = []

        for jti, session in self.sessions.items():
            if session["user_id"] == user_id:
                tokens_to_remove.append(jti)
                revoked += 1

        for jti in tokens_to_remove:
            del self.sessions[jti]

        logger.info(f"Revoked {revoked} tokens for user {user_id}")
        return revoked

    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions."""
        expired = 0
        now = datetime.utcnow()
        tokens_to_remove = []

        for jti, session in self.sessions.items():
            if now > session["expires_at"]:
                tokens_to_remove.append(jti)
                expired += 1

        for jti in tokens_to_remove:
            del self.sessions[jti]

        logger.info(f"Cleaned up {expired} expired sessions")
        return expired

    def get_session_info(self, token: str) -> Optional[Dict[str, Any]]:
        """Get information about current session."""
        payload = self.verify_token(token)
        if not payload:
            return None

        jti = payload.get("jti")
        if jti in self.sessions:
            session = self.sessions[jti].copy()
            session["user_id"] = payload["sub"]
            session["roles"] = [Role(role) for role in payload["roles"]]
            session["permissions"] = list(self.get_user_permissions(session["roles"]))
            return session

        return None

    def validate_resource_access(self,
                                 user_roles: List[Role],
                                 resource_type: str,
                                 action: str,
                                 resource_data: Optional[Dict] = None) -> bool:
        """Validate access to specific resource with custom logic."""
        # Base permission check
        permission_map = {
            ("transaction", "view"): Permission.VIEW_TRANSACTIONS,
            ("transaction", "create"): Permission.CREATE_TRANSACTIONS,
            ("transaction", "update"): Permission.UPDATE_TRANSACTIONS,
            ("fraud_case", "view"): Permission.VIEW_FRAUD_CASES,
            ("fraud_case", "create"): Permission.CREATE_FRAUD_CASES,
            ("policy", "view"): Permission.VIEW_POLICIES,
            ("user", "view"): Permission.VIEW_USERS,
        }

        permission = permission_map.get((resource_type, action))
        if permission and not self.has_permission(user_roles, permission):
            return False

        # Additional resource-specific checks
        if resource_type == "transaction" and resource_data:
            # Analysts can only view transactions from their assigned merchants
            if Role.ANALYST in user_roles:
                # This would check against user's assigned merchants
                pass

        return True


# Global instance
access_control = AccessControl()