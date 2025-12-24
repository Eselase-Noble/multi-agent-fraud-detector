import json
import re
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
import hashlib

from app.utils.logger import get_logger

logger = get_logger(__name__)


class PIIMasker:
    """Production-grade PII masking for financial data."""

    def __init__(self):
        # PII patterns
        self.patterns = {
            # Credit card numbers (16 digits, possibly with spaces/dashes)
            "credit_card": r'\b(?:\d[ -]*?){13,16}\b',

            # Email addresses
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',

            # Phone numbers (international and local formats)
            "phone": r'\b(?:\+\d{1,3}[-.]?)?\(?\d{3}\)?[-.]?\d{3}[-.]?\d{4}\b',

            # Social Security Numbers (US)
            "ssn": r'\b\d{3}[-]?\d{2}[-]?\d{4}\b',

            # Bank account numbers (generic pattern)
            "bank_account": r'\b\d{8,17}\b',

            # IP addresses
            "ip_address": r'\b(?:\d{1,3}\.){3}\d{1,3}\b',

            # Names (simple pattern - in production would use NER)
            "person_name": r'\b(?:Mr\.|Ms\.|Mrs\.|Dr\.)?\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b',

            # Addresses (simple pattern)
            "address": r'\b\d+\s+[A-Z][a-z]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd)\b',

            # Date of birth variations
            "dob": r'\b(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12][0-9]|3[01])[/-](?:19|20)\d{2}\b'
        }

        # Fields that commonly contain PII
        self.pii_fields = {
            'user_id', 'customer_id', 'account_number', 'card_number',
            'email', 'phone', 'address', 'birth_date', 'ssn',
            'first_name', 'last_name', 'full_name', 'ip_address',
            'driver_license', 'passport_number', 'tax_id'
        }

        # Masking strategies
        self.masking_strategies = {
            "hash": lambda x, salt: hashlib.sha256(f"{salt}{x}".encode()).hexdigest()[:16],
            "mask": lambda x, salt: "***MASKED***",
            "partial": self._partial_mask,
            "redact": lambda x, salt: "[REDACTED]"
        }

    def mask_text(self, text: str, strategy: str = "hash", salt: str = "") -> str:
        """Mask PII in text using specified strategy."""
        if not text or not isinstance(text, str):
            return text

        masked_text = text

        for pii_type, pattern in self.patterns.items():
            matches = list(re.finditer(pattern, masked_text, re.IGNORECASE))

            # Process matches in reverse to avoid position shifting
            for match in reversed(matches):
                pii_value = match.group(0)

                # Apply masking strategy
                if strategy in self.masking_strategies:
                    masked_value = self.masking_strategies[strategy](pii_value, salt)
                else:
                    masked_value = self.masking_strategies["hash"](pii_value, salt)

                # Replace in text
                start, end = match.span()
                masked_text = masked_text[:start] + masked_value + masked_text[end:]

        return masked_text

    def mask_dict(self, data: Dict[str, Any], strategy: str = "hash") -> Dict[str, Any]:
        """Mask PII in dictionary recursively."""
        if not data:
            return data

        masked_data = {}

        for key, value in data.items():
            if isinstance(value, dict):
                masked_data[key] = self.mask_dict(value, strategy)

            elif isinstance(value, list):
                masked_data[key] = [
                    self.mask_dict(item, strategy) if isinstance(item, dict)
                    else self.mask_text(str(item), strategy) if isinstance(item, str)
                    else item
                    for item in value
                ]

            elif isinstance(value, str):
                # Check if key indicates PII field
                if self._is_pii_field(key):
                    masked_data[key] = self.masking_strategies[strategy](value, key)
                else:
                    masked_data[key] = self.mask_text(value, strategy)

            else:
                masked_data[key] = value

        return masked_data

    def mask_transaction(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Mask PII in transaction data specifically."""
        if not transaction:
            return {}

        masked = transaction.copy()

        # Always mask these fields
        sensitive_fields = {
            'user_id': 'partial',
            'merchant_id': 'hash',
            'ip_address': 'hash',
            'user_country': 'keep',  # Country is usually okay
            'merchant_country': 'keep'
        }

        for field, strategy in sensitive_fields.items():
            if field in masked:
                if strategy == 'partial':
                    masked[field] = self._partial_mask(masked[field], field)
                elif strategy == 'hash':
                    masked[field] = self.masking_strategies['hash'](masked[field], field)
                # 'keep' means don't mask

        # Mask any PII in features field
        if 'features' in masked and isinstance(masked['features'], dict):
            masked['features'] = self.mask_dict(masked['features'])

        return masked

    def _partial_mask(self, value: str, salt: str = "") -> str:
        """Partially mask value (show first and last chars)."""
        if not value or len(value) <= 4:
            return "****"

        # Keep first 2 and last 2 characters
        first_part = value[:2]
        last_part = value[-2:]

        return f"{first_part}****{last_part}"

    def _is_pii_field(self, field_name: str) -> bool:
        """Check if field name indicates PII content."""
        field_lower = field_name.lower()

        # Check exact matches
        if field_lower in self.pii_fields:
            return True

        # Check partial matches
        for pii_field in self.pii_fields:
            if pii_field in field_lower or field_lower in pii_field:
                return True

        # Check common PII indicators
        pii_indicators = ['name', 'email', 'phone', 'address', 'id', 'number', 'ssn', 'dob']
        for indicator in pii_indicators:
            if indicator in field_lower:
                return True

        return False

    def detect_pii(self, text: str) -> List[Dict[str, Any]]:
        """Detect PII in text without masking."""
        if not text:
            return []

        detected = []

        for pii_type, pattern in self.patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)

            for match in matches:
                detected.append({
                    "type": pii_type,
                    "value": match.group(0),
                    "start": match.start(),
                    "end": match.end()
                })

        return detected

    def validate_masking(self, original: Dict[str, Any], masked: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that PII was properly masked."""
        validation = {
            "total_fields": 0,
            "masked_fields": 0,
            "unmasked_pii": [],
            "validation_passed": True
        }

        def compare_dicts(orig_dict, mask_dict, path=""):
            for key, orig_value in orig_dict.items():
                full_path = f"{path}.{key}" if path else key
                validation["total_fields"] += 1

                if key in mask_dict:
                    mask_value = mask_dict[key]

                    if isinstance(orig_value, dict) and isinstance(mask_value, dict):
                        compare_dicts(orig_value, mask_value, full_path)

                    elif isinstance(orig_value, str) and isinstance(mask_value, str):
                        # Check if this field should be masked
                        if self._is_pii_field(key):
                            if orig_value == mask_value:
                                validation["unmasked_pii"].append({
                                    "field": full_path,
                                    "original": orig_value,
                                    "masked": mask_value
                                })
                                validation["validation_passed"] = False
                            else:
                                validation["masked_fields"] += 1

        compare_dicts(original, masked)

        return validation

    def create_audit_trail(self,
                           original_data: Dict[str, Any],
                           masked_data: Dict[str, Any],
                           user_id: str,
                           action: str) -> Dict[str, Any]:
        """Create audit trail for PII masking."""
        validation = self.validate_masking(original_data, masked_data)

        return {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "action": action,
            "validation_result": validation,
            "original_data_hash": hashlib.sha256(
                json.dumps(original_data, sort_keys=True).encode()
            ).hexdigest(),
            "masked_data_hash": hashlib.sha256(
                json.dumps(masked_data, sort_keys=True).encode()
            ).hexdigest()
        }


# Global instance
pii_masker = PIIMasker()