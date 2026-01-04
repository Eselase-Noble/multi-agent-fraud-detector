# app/init.py - Create this new file
import asyncio
from typing import Dict, Any
import sys

from app.utils.logger import get_logger
from app.utils.config import settings
from app.db.postgres import db_manager

logger = get_logger(__name__)


class ServiceInitializer:
    """Initialize all services with proper dependency order."""

    def __init__(self):
        self.initialized = False
        self.services = {}

    async def initialize_all(self) -> Dict[str, Any]:
        """Initialize all services in correct order."""
        if self.initialized:
            return self.services

        logger.info("🚀 Initializing IntelliFraud Copilot services...")

        try:
            # 1. Initialize PostgreSQL Database

            await db_manager.initialize()
            self.services["postgres"] = db_manager
            logger.info("✅ PostgreSQL initialized")

            # 2. Initialize Vector Store
            from app.db.vector_store import vector_store
            await vector_store.initialize()
            self.services["vector_store"] = vector_store
            logger.info("✅ Vector store initialized")

            # 3. Initialize Audit Logger
            from app.security.audit_log import audit_logger
            await audit_logger.initialize()
            self.services["audit_logger"] = audit_logger
            logger.info("✅ Audit logger initialized")

            # 4. Seed sample data (if needed)
            if settings.DEBUG:
                from app.db.seed_data import DataSeeder
                seeder = DataSeeder()
                await seeder.initialize()
                await seeder.seed_transactions(20)
                await seeder.seed_policy_documents()
                logger.info("✅ Sample data seeded")

            # 5. Initialize LLM Clients
            from app.models.openai_client import openai_client
            from app.models.deepseek_client import deepseek_client

            # Test OpenAI connection
            try:
                await openai_client.validate_api_key()
                logger.info("✅ OpenAI client initialized")
            except Exception as e:
                logger.warning(f"⚠️ OpenAI client issue: {e}")

            # Test DeepSeek connection
            try:
                await deepseek_client.validate_api_key()
                logger.info("✅ DeepSeek client initialized")
            except Exception as e:
                logger.warning(f"⚠️ DeepSeek client issue: {e}")

            self.initialized = True
            logger.info("🎉 All services initialized successfully!")

            return self.services

        except Exception as e:
            logger.error(f"❌ Service initialization failed: {e}")
            raise

    async def close_all(self):
        """Close all services."""
        logger.info("Shutting down services...")

        # Close in reverse order
        try:
            from app.models.deepseek_client import deepseek_client
            await deepseek_client.close()
            logger.info("✅ DeepSeek client closed")
        except Exception as e:
            logger.error(f"Error closing DeepSeek: {e}")

        try:
            from app.db.postgres import db_manager
            await db_manager.close()
            logger.info("✅ PostgreSQL closed")
        except Exception as e:
            logger.error(f"Error closing PostgreSQL: {e}")

        self.initialized = False
        logger.info("All services closed")


# Global instance
initializer = ServiceInitializer()