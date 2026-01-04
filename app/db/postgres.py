import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
from contextlib import contextmanager, asynccontextmanager

import psycopg2
from psycopg2.extras import RealDictCursor, Json
from psycopg2.pool import SimpleConnectionPool
import pandas as pd

from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class PostgreSQLManager:
    """Production-grade PostgreSQL database manager."""

    def __init__(self):
        self.connection_pool = None
        self.pool_size = 20
        self.max_overflow = 10

    async def initialize(self):
        """Initialize connection pool."""
        try:
            self.connection_pool = SimpleConnectionPool(
                minconn=1,
                maxconn=self.pool_size + self.max_overflow,
                dsn=settings.DATABASE_URL
            )

            # ✅ Direct pool usage for startup test
            conn = self.connection_pool.getconn()
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT version();")
                    version = cur.fetchone()
                    logger.info(f"PostgreSQL connected: {version[0]}")
            finally:
                self.connection_pool.putconn(conn)

            logger.info(
                f"PostgreSQL connection pool initialized "
                f"(size: {self.pool_size + self.max_overflow})"
            )

        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL: {e}")
            self.connection_pool = None
            raise

    @contextmanager
    def _get_connection(self):
        if self.connection_pool is None:
            raise RuntimeError(
                "PostgreSQL connection pool is not initialized. "
                "Did you forget to await init_db() on startup?"
            )

        conn = None
        try:
            conn = self.connection_pool.getconn()
            yield conn
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                self.connection_pool.putconn(conn)

    @asynccontextmanager
    async def get_connection(self):
        if self.connection_pool is None:
            raise RuntimeError(
                "PostgreSQL connection pool is not initialized. Did you forget to await init_db() on startup?"
            )
        conn = None
        try:
            conn = self.connection_pool.getconn()
            yield conn
        finally:
            if conn:
                self.connection_pool.putconn(conn)

    @contextmanager
    def _get_cursor(self, conn, cursor_factory=None):
        """Get cursor with context management."""
        cursor = None
        try:
            if cursor_factory:
                cursor = conn.cursor(cursor_factory=cursor_factory)
            else:
                cursor = conn.cursor()
            yield cursor
        finally:
            if cursor:
                cursor.close()

    async def get_transaction_data(self,
                                   transaction_id: Optional[str] = None,
                                   user_id: Optional[str] = None,
                                   start_time: Optional[datetime] = None,
                                   end_time: Optional[datetime] = None,
                                   limit: int = 100) -> List[Dict[str, Any]]:
        """Get transaction data with flexible filtering."""
        query = """
            SELECT 
                transaction_id,
                user_id,
                amount,
                currency,
                merchant_id,
                merchant_category,
                merchant_country,
                user_country,
                transaction_time,
                status,
                fraud_score,
                features
            FROM transactions
            WHERE 1=1
        """

        params = []
        param_count = 0

        if transaction_id:
            param_count += 1
            query += f" AND transaction_id = %s"
            params.append(transaction_id)

        if user_id:
            param_count += 1
            query += f" AND user_id = %s"
            params.append(user_id)

        if start_time:
            param_count += 1
            query += f" AND transaction_time >= %s"
            params.append(start_time)

        if end_time:
            param_count += 1
            query += f" AND transaction_time <= %s"
            params.append(end_time)

        query += f" ORDER BY transaction_time DESC LIMIT %s"
        params.append(limit)

        try:
            with self._get_connection() as conn:
                with self._get_cursor(conn, RealDictCursor) as cur:
                    cur.execute(query, params)
                    results = cur.fetchall()

                    # Convert to list of dicts
                    transactions = []
                    for row in results:
                        transaction = dict(row)

                        # Parse JSON fields
                        if transaction.get('features'):
                            try:
                                transaction['features'] = json.loads(transaction['features'])
                            except:
                                transaction['features'] = {}

                        transactions.append(transaction)

                    logger.debug(f"Retrieved {len(transactions)} transactions")
                    return transactions

        except Exception as e:
            logger.error(f"Failed to get transaction data: {e}")
            return []

    async def get_user_transaction_history(self,
                                           user_id: str,
                                           days: int = 30) -> List[Dict[str, Any]]:
        """Get user's transaction history."""
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        return await self.get_transaction_data(
            user_id=user_id,
            start_time=start_time,
            end_time=end_time,
            limit=1000
        )

    async def get_transaction_velocity(self,
                                       user_id: str,
                                       hours: int = 24) -> Dict[str, Any]:
        """Calculate transaction velocity for a user."""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)

        query = """
            SELECT 
                COUNT(*) as count,
                SUM(amount) as total_amount,
                AVG(amount) as avg_amount,
                MIN(transaction_time) as first_transaction,
                MAX(transaction_time) as last_transaction
            FROM transactions
            WHERE user_id = %s 
              AND transaction_time >= %s 
              AND transaction_time <= %s
        """

        try:
            with self._get_connection() as conn:
                with self._get_cursor(conn, RealDictCursor) as cur:
                    cur.execute(query, (user_id, start_time, end_time))
                    result = cur.fetchone()

                    velocity = {
                        "count": result['count'] if result['count'] else 0,
                        "total_amount": float(result['total_amount'] or 0),
                        "avg_amount": float(result['avg_amount'] or 0),
                        "period_hours": hours,
                        "transactions_per_hour": result['count'] / hours if hours > 0 else 0,
                        "first_transaction": result['first_transaction'],
                        "last_transaction": result['last_transaction']
                    }

                    return velocity

        except Exception as e:
            logger.error(f"Failed to calculate transaction velocity: {e}")
            return {}

    async def get_merchant_risk_data(self, merchant_id: str) -> Dict[str, Any]:
        """Get risk data for a merchant."""
        query = """
            SELECT 
                merchant_id,
                merchant_category,
                merchant_country,
                COUNT(*) as total_transactions,
                SUM(CASE WHEN fraud_score > 0.7 THEN 1 ELSE 0 END) as high_risk_count,
                AVG(fraud_score) as avg_fraud_score,
                AVG(amount) as avg_transaction_amount
            FROM transactions
            WHERE merchant_id = %s
            GROUP BY merchant_id, merchant_category, merchant_country
        """

        try:
            with self._get_connection() as conn:
                with self._get_cursor(conn, RealDictCursor) as cur:
                    cur.execute(query, (merchant_id,))
                    result = cur.fetchone()

                    if result:
                        risk_data = {
                            "merchant_id": result['merchant_id'],
                            "category": result['merchant_category'],
                            "country": result['merchant_country'],
                            "total_transactions": result['total_transactions'],
                            "high_risk_transactions": result['high_risk_count'],
                            "high_risk_percentage": (result['high_risk_count'] / result['total_transactions']) * 100
                            if result['total_transactions'] > 0 else 0,
                            "avg_fraud_score": float(result['avg_fraud_score'] or 0),
                            "avg_transaction_amount": float(result['avg_transaction_amount'] or 0)
                        }
                        return risk_data

                    return {}

        except Exception as e:
            logger.error(f"Failed to get merchant risk data: {e}")
            return {}

    async def insert_transaction(self, transaction_data: Dict[str, Any]) -> str:
        """Insert a new transaction."""
        query = """
            INSERT INTO transactions (
                transaction_id, user_id, amount, currency,
                merchant_id, merchant_category, merchant_country,
                user_country, transaction_time, status,
                fraud_score, features
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING transaction_id
        """

        params = (
            transaction_data.get('transaction_id'),
            transaction_data.get('user_id'),
            transaction_data.get('amount'),
            transaction_data.get('currency'),
            transaction_data.get('merchant_id'),
            transaction_data.get('merchant_category'),
            transaction_data.get('merchant_country'),
            transaction_data.get('user_country'),
            transaction_data.get('transaction_time', datetime.now()),
            transaction_data.get('status', 'pending'),
            transaction_data.get('fraud_score', 0.0),
            Json(transaction_data.get('features', {}))
        )

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                    transaction_id = cur.fetchone()[0]
                    conn.commit()

                    logger.info(f"Inserted transaction: {transaction_id}")
                    return transaction_id

        except Exception as e:
            logger.error(f"Failed to insert transaction: {e}")
            raise

    async def update_transaction_fraud_score(self,
                                             transaction_id: str,
                                             fraud_score: float) -> bool:
        """Update fraud score for transaction."""
        query = """
            UPDATE transactions
            SET fraud_score = %s,
                status = CASE 
                    WHEN %s > 0.8 THEN 'flagged' 
                    WHEN %s > 0.6 THEN 'suspicious' 
                    ELSE status 
                END
            WHERE transaction_id = %s
        """

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, (fraud_score, fraud_score, fraud_score, transaction_id))
                    rows_affected = cur.rowcount
                    conn.commit()

                    success = rows_affected > 0
                    if success:
                        logger.info(f"Updated fraud score for {transaction_id}: {fraud_score}")

                    return success

        except Exception as e:
            logger.error(f"Failed to update fraud score: {e}")
            return False

    async def get_fraud_cases(self,
                              fraud_type: Optional[str] = None,
                              status: Optional[str] = None,
                              limit: int = 50) -> List[Dict[str, Any]]:
        """Get fraud cases with filtering."""
        query = """
            SELECT 
                case_id,
                transaction_ids,
                fraud_type,
                description,
                amount_lost,
                detection_method,
                created_at,
                resolved_at,
                status
            FROM fraud_cases
            WHERE 1=1
        """

        params = []

        if fraud_type:
            query += " AND fraud_type = %s"
            params.append(fraud_type)

        if status:
            query += " AND status = %s"
            params.append(status)

        query += " ORDER BY created_at DESC LIMIT %s"
        params.append(limit)

        try:
            with self._get_connection() as conn:
                with self._get_cursor(conn, RealDictCursor) as cur:
                    cur.execute(query, params)
                    results = cur.fetchall()

                    fraud_cases = []
                    for row in results:
                        case = dict(row)

                        # Parse array fields
                        if case.get('transaction_ids'):
                            case['transaction_ids'] = list(case['transaction_ids'])

                        fraud_cases.append(case)

                    return fraud_cases

        except Exception as e:
            logger.error(f"Failed to get fraud cases: {e}")
            return []

    async def create_fraud_case(self, case_data: Dict[str, Any]) -> int:
        """Create a new fraud case."""
        query = """
            INSERT INTO fraud_cases (
                transaction_ids,
                fraud_type,
                description,
                amount_lost,
                detection_method,
                status
            ) VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING case_id
        """

        params = (
            case_data.get('transaction_ids', []),
            case_data.get('fraud_type', 'unknown'),
            case_data.get('description', ''),
            case_data.get('amount_lost', 0.0),
            case_data.get('detection_method', 'system'),
            case_data.get('status', 'open')
        )

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                    case_id = cur.fetchone()[0]
                    conn.commit()

                    logger.info(f"Created fraud case: {case_id}")
                    return case_id

        except Exception as e:
            logger.error(f"Failed to create fraud case: {e}")
            raise

    async def get_policy_documents(self,
                                   policy_type: Optional[str] = None,
                                   region: Optional[str] = None,
                                   limit: int = 100) -> List[Dict[str, Any]]:
        """Get policy documents."""
        query = """
            SELECT 
                policy_id,
                title,
                content,
                policy_type,
                region,
                version,
                effective_date,
                created_at
            FROM policies
            WHERE 1=1
        """

        params = []

        if policy_type:
            query += " AND policy_type = %s"
            params.append(policy_type)

        if region:
            query += " AND region = %s"
            params.append(region)

        query += " ORDER BY effective_date DESC LIMIT %s"
        params.append(limit)

        try:
            with self._get_connection() as conn:
                with self._get_cursor(conn, RealDictCursor) as cur:
                    cur.execute(query, params)
                    results = cur.fetchall()
                    return [dict(row) for row in results]

        except Exception as e:
            logger.error(f"Failed to get policy documents: {e}")
            return []

    async def log_audit_event(self, audit_data: Dict[str, Any]) -> int:
        """Log an audit event."""
        query = """
            INSERT INTO audit_logs (
                user_id,
                action,
                resource_type,
                resource_id,
                request_body,
                response_body,
                status_code,
                ip_address,
                user_agent
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING log_id
        """

        params = (
            audit_data.get('user_id'),
            audit_data.get('action'),
            audit_data.get('resource_type'),
            audit_data.get('resource_id'),
            Json(audit_data.get('request_body', {})),
            Json(audit_data.get('response_body', {})),
            audit_data.get('status_code'),
            audit_data.get('ip_address'),
            audit_data.get('user_agent')
        )

        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                    log_id = cur.fetchone()[0]
                    conn.commit()

                    logger.debug(f"Logged audit event: {log_id}")
                    return log_id

        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")
            raise

    async def get_system_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """Get system performance metrics."""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)

        metrics = {}

        # Transaction metrics
        tx_query = """
            SELECT 
                COUNT(*) as total_transactions,
                SUM(CASE WHEN fraud_score > 0.7 THEN 1 ELSE 0 END) as high_risk_transactions,
                AVG(fraud_score) as avg_fraud_score,
                COUNT(DISTINCT user_id) as unique_users,
                COUNT(DISTINCT merchant_id) as unique_merchants
            FROM transactions
            WHERE transaction_time >= %s AND transaction_time <= %s
        """

        # Fraud case metrics
        fraud_query = """
            SELECT 
                COUNT(*) as total_cases,
                COUNT(CASE WHEN status = 'open' THEN 1 END) as open_cases,
                COUNT(CASE WHEN status = 'resolved' THEN 1 END) as resolved_cases,
                SUM(amount_lost) as total_amount_lost
            FROM fraud_cases
            WHERE created_at >= %s AND created_at <= %s
        """

        try:
            with self._get_connection() as conn:
                # Get transaction metrics
                with self._get_cursor(conn, RealDictCursor) as cur:
                    cur.execute(tx_query, (start_time, end_time))
                    tx_metrics = cur.fetchone()
                    metrics['transactions'] = dict(tx_metrics) if tx_metrics else {}

                # Get fraud case metrics
                with self._get_cursor(conn, RealDictCursor) as cur:
                    cur.execute(fraud_query, (start_time, end_time))
                    fraud_metrics = cur.fetchone()
                    metrics['fraud_cases'] = dict(fraud_metrics) if fraud_metrics else {}

                # Calculate rates
                if metrics['transactions'].get('total_transactions', 0) > 0:
                    metrics['fraud_rate'] = (
                            metrics['transactions']['high_risk_transactions'] /
                            metrics['transactions']['total_transactions'] * 100
                    )
                else:
                    metrics['fraud_rate'] = 0.0

                metrics['time_period'] = {
                    'start': start_time,
                    'end': end_time,
                    'hours': hours
                }

                return metrics

        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return {}

    async def execute_custom_query(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """Execute custom SQL query."""
        try:
            with self._get_connection() as conn:
                with self._get_cursor(conn, RealDictCursor) as cur:
                    cur.execute(query, params)
                    results = cur.fetchall()
                    return [dict(row) for row in results]

        except Exception as e:
            logger.error(f"Failed to execute custom query: {e}")
            raise

    async def close(self):
        """Close all database connections."""
        if self.connection_pool:
            self.connection_pool.closeall()
            logger.info("PostgreSQL connections closed")


# Global instance
db_manager = PostgreSQLManager()


async def init_db():
    """Initialize database connection."""
    await db_manager.initialize()


async def close_db():
    """Close database connection."""
    await db_manager.close()


# async def get_transaction_data(transaction_id: str) -> Optional[Dict[str, Any]]:
#     """Get single transaction by ID."""
#     transactions = await db_manager.get_transaction_data(transaction_id=transaction_id, limit=1)
#     return transactions[0] if transactions else None

# FIXED (Correct):
async def get_transaction_data(transaction_id: str = None,
                             user_id: str = None,
                             start_time: Optional[datetime] = None,
                             end_time: Optional[datetime] = None,
                             limit: int = 100) -> Optional[List[Dict[str, Any]]]:
    """Get transaction data with flexible filtering."""
    return await db_manager.get_transaction_data(
        transaction_id=transaction_id,
        user_id=user_id,
        start_time=start_time,
        end_time=end_time,
        limit=limit
    )

async def get_single_transaction(transaction_id: str) -> Optional[Dict[str, Any]]:
    """Get single transaction by ID."""
    transactions = await get_transaction_data(transaction_id=transaction_id, limit=1)
    return transactions[0] if transactions else None