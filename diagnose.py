# diagnose.py
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


async def diagnose():
    print("🔍 Diagnosing IntelliFraud Copilot...")

    # Check 1: Is PostgreSQL running?
    print("\n1. Checking PostgreSQL...")
    try:
        import psycopg2
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="fraud_db",
            user="postgres",
            password="postgres"
        )
        conn.close()
        print("   ✅ PostgreSQL is running")
    except Exception as e:
        print(f"   ❌ PostgreSQL connection failed: {e}")
        print("   Run: docker-compose up -d db")
        return False

    # Check 2: Database tables
    print("\n2. Checking database tables...")
    try:
        import psycopg2
        conn = psycopg2.connect(
            host="localhost",
            port=5432,
            database="fraud_db",
            user="rag_user",
            password="RagUser12/."
        )
        cursor = conn.cursor()
        cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
        tables = cursor.fetchall()
        print(f"   ✅ Found {len(tables)} tables")
        for table in tables[:5]:
            print(f"     - {table[0]}")
        conn.close()
    except Exception as e:
        print(f"   ❌ Error checking tables: {e}")

    # Check 3: Initialize db_manager
    print("\n3. Testing db_manager initialization...")
    try:
        from app.db.postgres import db_manager
        print(f"   db_manager type: {type(db_manager)}")
        print(f"   connection_pool: {db_manager.connection_pool}")

        if db_manager.connection_pool is None:
            print("   ⚠️ connection_pool is None, initializing...")
            await db_manager.initialize()
            print(f"   After init, connection_pool: {db_manager.connection_pool}")
        else:
            print("   ✅ db_manager already initialized")
    except Exception as e:
        print(f"   ❌ db_manager error: {e}")
        return False

    print("\n🎯 Diagnosis complete!")
    return True


if __name__ == "__main__":
    success = asyncio.run(diagnose())
    sys.exit(0 if success else 1)