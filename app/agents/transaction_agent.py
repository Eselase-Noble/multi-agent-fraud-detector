import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass

from app.db.postgres import get_transaction_data
from app.models.deepseek_client import DeepSeekClient
from app.utils.logger import get_logger
from app.utils.config import settings


#author: Noble Eselase Vulley
#version: 1.0.0


logger = get_logger(__name__)


@dataclass
class TransactionFeatures:
    """Extracted features from transaction data."""
    amount: float
    velocity_1h: float
    velocity_24h: float
    velocity_7d: float
    geo_distance_km: Optional[float]
    merchant_risk_score: float
    time_of_day: float
    day_of_week: int
    is_weekend: bool
    amount_deviation: float
    category_risk: float
    country_risk: float


@dataclass
class AnomalyScores:
    """Anomaly scores for different dimensions."""
    velocity: float
    geo: float
    amount: float
    time: float
    merchant: float
    behavioral: float


class TransactionAgent:
    def __init__(self):
        self.deepseek_client = DeepSeekClient()
        self.system_prompt = """You are a Transaction Analysis Agent.

        Analyze the provided transaction data and compute:
        - Velocity anomalies (transactions per hour/day)
        - Geo-location inconsistencies
        - Merchant risk indicators
        - Historical fraud similarity
        - Behavioral patterns

        Return JSON only with this structure:
        {
            "transaction_id": "id",
            "analysis": {
                "velocity_anomaly": {
                    "score": 0.85,
                    "description": "5x normal velocity",
                    "confidence": 0.9
                },
                "geo_anomaly": {
                    "score": 0.0,
                    "description": "No anomaly",
                    "confidence": 0.8
                },
                "merchant_risk": {
                    "score": 0.6,
                    "description": "Medium risk merchant",
                    "confidence": 0.7
                },
                "behavioral_anomaly": {
                    "score": 0.75,
                    "description": "Unusual spending pattern",
                    "confidence": 0.85
                }
            },
            "features": {
                "amount": 1000.0,
                "velocity_1h": 3,
                "velocity_24h": 15,
                "geo_distance_km": 1500.0,
                "merchant_risk_score": 0.6
            },
            "overall_risk": 0.65,
            "confidence": 0.8,
            "recommendations": ["flag_for_review", "enhanced_monitoring"]
        }

        Transaction Data:
        {transaction_json}"""

        # Risk thresholds
        self.velocity_threshold = 5  # 5x normal velocity
        self.geo_threshold_km = 500  # 500km from usual location
        self.amount_threshold = 3.0  # 3x normal amount
        self.merchant_risk_threshold = 0.7

    async def analyze_transaction(self, transaction_id: str) -> Dict[str, Any]:
        """Analyze a single transaction for fraud indicators."""
        try:
            logger.info(f"Analyzing transaction: {transaction_id}")

            # Get transaction data
            transaction_data = await get_transaction_data(transaction_id)
            if not transaction_data:
                return self._create_empty_analysis(transaction_id)

            # Extract features
            features = await self._extract_features(transaction_data)

            # Calculate anomaly scores
            anomaly_scores = await self._calculate_anomalies(features, transaction_data)

            # Get historical context
            historical_context = await self._get_historical_context(
                transaction_data["user_id"],
                transaction_data["transaction_time"]
            )

            # Use DeepSeek for pattern analysis
            llm_analysis = await self._llm_analysis(
                transaction_data,
                features,
                anomaly_scores,
                historical_context
            )

            # Combine results
            analysis = self._combine_analysis(
                transaction_id,
                transaction_data,
                features,
                anomaly_scores,
                llm_analysis,
                historical_context
            )

            logger.info(f"Transaction {transaction_id} analysis complete: risk={analysis['overall_risk']:.2f}")
            return analysis

        except Exception as e:
            logger.error(f"Failed to analyze transaction {transaction_id}: {e}", exc_info=True)
            return self._create_error_analysis(transaction_id, str(e))

    async def analyze_transaction_batch(self, transaction_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Analyze multiple transactions in batch."""
        tasks = [self.analyze_transaction(tx_id) for tx_id in transaction_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert to dictionary
        analysis_dict = {}
        for tx_id, result in zip(transaction_ids, results):
            if isinstance(result, Exception):
                analysis_dict[tx_id] = self._create_error_analysis(tx_id, str(result))
            else:
                analysis_dict[tx_id] = result

        # Add cross-transaction analysis
        if len(transaction_ids) > 1:
            cross_analysis = await self._cross_transaction_analysis(analysis_dict)
            analysis_dict["_cross_analysis"] = cross_analysis

        return analysis_dict

    async def _extract_features(self, transaction: Dict) -> TransactionFeatures:
        """Extract features from transaction data."""
        # Get user's recent transactions for velocity calculation
        recent_transactions = await get_transaction_data(
            user_id=transaction["user_id"],
            start_time=transaction["transaction_time"] - timedelta(days=7)
        )

        # Calculate velocities
        velocity_1h = self._calculate_velocity(recent_transactions, hours=1)
        velocity_24h = self._calculate_velocity(recent_transactions, hours=24)
        velocity_7d = self._calculate_velocity(recent_transactions, hours=168)

        # Calculate geo distance
        geo_distance = await self._calculate_geo_distance(
            transaction["user_country"],
            transaction["merchant_country"],
            transaction.get("user_location"),
            transaction.get("merchant_location")
        )

        # Calculate merchant risk
        merchant_risk = await self._calculate_merchant_risk(
            transaction["merchant_id"],
            transaction["merchant_category"]
        )

        # Calculate amount deviation
        amount_deviation = self._calculate_amount_deviation(
            transaction["amount"],
            recent_transactions
        )

        # Time features
        tx_time = transaction["transaction_time"]
        time_of_day = tx_time.hour + tx_time.minute / 60.0
        day_of_week = tx_time.weekday()
        is_weekend = day_of_week >= 5

        return TransactionFeatures(
            amount=transaction["amount"],
            velocity_1h=velocity_1h,
            velocity_24h=velocity_24h,
            velocity_7d=velocity_7d,
            geo_distance_km=geo_distance,
            merchant_risk_score=merchant_risk,
            time_of_day=time_of_day,
            day_of_week=day_of_week,
            is_weekend=is_weekend,
            amount_deviation=amount_deviation,
            category_risk=self._category_risk_score(transaction["merchant_category"]),
            country_risk=self._country_risk_score(transaction["merchant_country"])
        )

    async def _calculate_anomalies(self, features: TransactionFeatures, transaction: Dict) -> AnomalyScores:
        """Calculate anomaly scores for different dimensions."""
        # Velocity anomaly
        normal_velocity = features.velocity_24h / 24  # Per hour
        velocity_ratio = features.velocity_1h / normal_velocity if normal_velocity > 0 else 0
        velocity_score = min(1.0, velocity_ratio / self.velocity_threshold)

        # Geo anomaly
        geo_score = 0.0
        if features.geo_distance_km:
            geo_score = min(1.0, features.geo_distance_km / self.geo_threshold_km)

        # Amount anomaly
        amount_score = min(1.0, features.amount_deviation / self.amount_threshold)

        # Time anomaly
        time_score = self._calculate_time_anomaly(features.time_of_day, features.day_of_week)

        # Merchant risk
        merchant_score = features.merchant_risk_score

        # Behavioral anomaly (combination)
        behavioral_score = np.mean([
            velocity_score * 0.3,
            geo_score * 0.2,
            amount_score * 0.2,
            time_score * 0.1,
            merchant_score * 0.2
        ])

        return AnomalyScores(
            velocity=velocity_score,
            geo=geo_score,
            amount=amount_score,
            time=time_score,
            merchant=merchant_score,
            behavioral=behavioral_score
        )

    async def _llm_analysis(self, transaction: Dict, features: TransactionFeatures,
                            anomalies: AnomalyScores, historical: Dict) -> Dict[str, Any]:
        """Use LLM for pattern recognition and analysis."""
        analysis_context = {
            "transaction": transaction,
            "features": features.__dict__,
            "anomalies": anomalies.__dict__,
            "historical": historical
        }

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": json.dumps(analysis_context, default=str)}
        ]

        response = await self.deepseek_client.chat_completion(
            messages=messages,
            temperature=0.1,
            max_tokens=500
        )

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.warning("LLM returned invalid JSON, using fallback analysis")
            return self._create_fallback_analysis(transaction, anomalies)

    def _combine_analysis(self, transaction_id: str, transaction: Dict,
                          features: TransactionFeatures, anomalies: AnomalyScores,
                          llm_analysis: Dict, historical: Dict) -> Dict[str, Any]:
        """Combine all analysis components."""
        # Calculate overall risk (weighted average)
        weights = {
            'velocity': 0.25,
            'geo': 0.20,
            'amount': 0.15,
            'merchant': 0.20,
            'behavioral': 0.20
        }

        overall_risk = (
                anomalies.velocity * weights['velocity'] +
                anomalies.geo * weights['geo'] +
                anomalies.amount * weights['amount'] +
                anomalies.merchant * weights['merchant'] +
                anomalies.behavioral * weights['behavioral']
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(anomalies, overall_risk)

        # Format response
        return {
            "transaction_id": transaction_id,
            "analysis": {
                "velocity_anomaly": {
                    "score": float(anomalies.velocity),
                    "description": self._velocity_description(anomalies.velocity),
                    "confidence": 0.9
                },
                "geo_anomaly": {
                    "score": float(anomalies.geo),
                    "description": self._geo_description(anomalies.geo),
                    "confidence": 0.8
                },
                "merchant_risk": {
                    "score": float(anomalies.merchant),
                    "description": self._merchant_description(anomalies.merchant),
                    "confidence": 0.7
                },
                "behavioral_anomaly": {
                    "score": float(anomalies.behavioral),
                    "description": self._behavioral_description(anomalies.behavioral),
                    "confidence": 0.85
                }
            },
            "features": features.__dict__,
            "llm_insights": llm_analysis.get("analysis", {}),
            "historical_context": historical,
            "overall_risk": float(overall_risk),
            "risk_level": self._risk_level(overall_risk),
            "confidence": llm_analysis.get("confidence", 0.8),
            "recommendations": recommendations,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": {
                "model_used": "deepseek",
                "analysis_version": "1.0",
                "processing_time_ms": 0  # Would be calculated in production
            }
        }

    # Helper methods (abbreviated for space)
    def _calculate_velocity(self, transactions: List[Dict], hours: int) -> float:
        if not transactions:
            return 0.0
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        recent = [t for t in transactions if t["transaction_time"] > cutoff]
        return len(recent) / hours

    async def _calculate_geo_distance(self, user_country: str, merchant_country: str,
                                      user_loc: Optional[Tuple], merchant_loc: Optional[Tuple]) -> Optional[float]:
        if user_loc and merchant_loc:
            # Calculate actual distance
            from geopy.distance import geodesic
            return geodesic(user_loc, merchant_loc).km
        elif user_country != merchant_country:
            # Country-level distance approximation
            return 500.0  # Default for cross-country
        return None

    async def _calculate_merchant_risk(self, merchant_id: str, category: str) -> float:
        # In production, this would query a merchant risk database
        risky_categories = {'gambling', 'adult', 'cryptocurrency', 'travel'}
        if category.lower() in risky_categories:
            return 0.8
        return 0.3

    def _calculate_amount_deviation(self, amount: float, history: List[Dict]) -> float:
        if not history:
            return 1.0
        amounts = [t["amount"] for t in history]
        mean = np.mean(amounts)
        std = np.std(amounts) if len(amounts) > 1 else mean * 0.5
        if std == 0:
            return 0.0
        return abs(amount - mean) / std

    def _calculate_time_anomaly(self, time_of_day: float, day_of_week: int) -> float:
        # Normal spending hours: 8 AM to 10 PM
        if 8 <= time_of_day <= 22:
            return 0.1
        return 0.8

    async def _get_historical_context(self, user_id: str, timestamp: datetime) -> Dict:
        # Get user's transaction history
        history = await get_transaction_data(
            user_id=user_id,
            end_time=timestamp - timedelta(hours=1),
            limit=100
        )

        return {
            "total_transactions": len(history),
            "avg_amount": np.mean([t["amount"] for t in history]) if history else 0,
            "common_categories": self._most_common_categories(history),
            "fraud_history": await self._get_user_fraud_history(user_id),
            "recent_velocity": self._calculate_velocity(history, hours=24) if history else 0
        }

    async def _cross_transaction_analysis(self, analyses: Dict[str, Dict]) -> Dict[str, Any]:
        """Analyze patterns across multiple transactions."""
        risk_scores = [a["overall_risk"] for a in analyses.values() if isinstance(a, dict)]

        return {
            "count": len(risk_scores),
            "avg_risk": float(np.mean(risk_scores)) if risk_scores else 0.0,
            "max_risk": float(np.max(risk_scores)) if risk_scores else 0.0,
            "risk_trend": "increasing" if len(risk_scores) > 1 and risk_scores[-1] > risk_scores[0] else "stable",
            "common_patterns": await self._detect_common_patterns(analyses)
        }

    def _generate_recommendations(self, anomalies: AnomalyScores, overall_risk: float) -> List[str]:
        recommendations = []

        if overall_risk > 0.8:
            recommendations.extend(["block_transaction", "alert_analyst", "freeze_account"])
        elif overall_risk > 0.6:
            recommendations.extend(["flag_for_review", "enhanced_monitoring", "request_2fa"])
        elif overall_risk > 0.4:
            recommendations.extend(["monitor_closely", "log_for_patterns"])

        if anomalies.velocity > 0.7:
            recommendations.append("check_velocity_limits")
        if anomalies.geo > 0.7:
            recommendations.append("verify_location")
        if anomalies.merchant > 0.7:
            recommendations.append("review_merchant_relationship")

        return list(set(recommendations))  # Remove duplicates

    def _create_empty_analysis(self, transaction_id: str) -> Dict[str, Any]:
        return {
            "transaction_id": transaction_id,
            "error": "Transaction not found",
            "analysis": {},
            "overall_risk": 0.0,
            "confidence": 0.0,
            "recommendations": ["verify_transaction_id"]
        }

    def _create_error_analysis(self, transaction_id: str, error: str) -> Dict[str, Any]:
        return {
            "transaction_id": transaction_id,
            "error": error,
            "analysis": {},
            "overall_risk": 0.5,  # Conservative risk score on error
            "confidence": 0.0,
            "recommendations": ["manual_review", "check_system_logs"]
        }

    def _create_fallback_analysis(self, transaction: Dict, anomalies: AnomalyScores) -> Dict[str, Any]:
        return {
            "analysis": {
                "velocity_anomaly": {"score": float(anomalies.velocity)},
                "geo_anomaly": {"score": float(anomalies.geo)},
                "merchant_risk": {"score": float(anomalies.merchant)}
            },
            "confidence": 0.6
        }

    def _velocity_description(self, score: float) -> str:
        if score > 0.8:
            return "Extremely high transaction velocity"
        elif score > 0.6:
            return "High transaction velocity"
        elif score > 0.4:
            return "Moderate transaction velocity"
        return "Normal transaction velocity"

    def _geo_description(self, score: float) -> str:
        if score > 0.8:
            return "Unusual geographic location"
        elif score > 0.6:
            return "Suspicious location change"
        return "Normal geographic pattern"

    def _merchant_description(self, score: float) -> str:
        if score > 0.8:
            return "High-risk merchant category"
        elif score > 0.6:
            return "Medium-risk merchant"
        return "Low-risk merchant"

    def _behavioral_description(self, score: float) -> str:
        if score > 0.8:
            return "Highly unusual behavioral pattern"
        elif score > 0.6:
            return "Suspicious behavioral pattern"
        elif score > 0.4:
            return "Slightly unusual behavior"
        return "Normal behavioral pattern"

    def _risk_level(self, score: float) -> str:
        if score > 0.8:
            return "CRITICAL"
        elif score > 0.6:
            return "HIGH"
        elif score > 0.4:
            return "MEDIUM"
        elif score > 0.2:
            return "LOW"
        return "NONE"

    def _category_risk_score(self, category: str) -> float:
        risk_categories = {
            'gambling': 0.9,
            'adult': 0.8,
            'cryptocurrency': 0.7,
            'travel': 0.4,
            'retail': 0.2,
            'groceries': 0.1
        }
        return risk_categories.get(category.lower(), 0.3)

    def _country_risk_score(self, country: str) -> float:
        high_risk_countries = {'RU', 'CN', 'NG', 'IR', 'KP'}
        if country in high_risk_countries:
            return 0.8
        return 0.2

    def _most_common_categories(self, transactions: List[Dict]) -> List[str]:
        from collections import Counter
        categories = [t.get("merchant_category", "unknown") for t in transactions]
        return [cat for cat, _ in Counter(categories).most_common(3)]

    async def _get_user_fraud_history(self, user_id: str) -> List[Dict]:
        # In production, query fraud cases database
        return []

    async def _detect_common_patterns(self, analyses: Dict) -> List[str]:
        # Pattern detection logic
        return []