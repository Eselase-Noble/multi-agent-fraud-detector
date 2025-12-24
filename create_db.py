# run_init_sql.py
import asyncpg
import asyncio
from urllib.parse import urlparse
from app.utils.config import settings
import os


def parse_db_url(db_url: str):
    """Parse DATABASE_URL into components."""
    parsed = urlparse(db_url)
    return {
        'host':  'localhost',
        'port': 5432,
        'user':'rag_user',
        'password': 'RagUser12/.',
        'database':  'fraud_db'
    }


async def run_init_sql():
    # Parse the DATABASE_URL
    db_config = parse_db_url(settings.DATABASE_URL)
    target_db = db_config['database']

    print(f"🔗 Connecting to database: {target_db}")

    # Find init.sql
    sql_file = 'init.sql'

    if not os.path.exists(sql_file):
        print(f"❌ ERROR: {sql_file} not found in current directory!")
        print(f"📁 Current directory: {os.getcwd()}")
        print("\nLooking for init.sql in other locations...")

        # Search for init.sql
        import glob
        sql_files = glob.glob('**/init.sql', recursive=True)

        if sql_files:
            print(f"📄 Found init.sql files: {sql_files}")
            sql_file = sql_files[0]
            print(f"📄 Using: {sql_file}")
        else:
            print("❌ No init.sql found anywhere!")
            print("💡 Please make sure init.sql exists in your project.")
            return

    print(f"📄 Reading SQL from: {sql_file}")

    # Read the SQL file
    with open(sql_file, 'r') as f:
        sql_commands = f.read()

    print(f"📏 SQL file size: {len(sql_commands)} characters")

    # Connect to the database
    try:
        conn = await asyncpg.connect(
            host=db_config['host'],
            port=db_config['port'],
            user=db_config['user'],
            password=db_config['password'],
            database=target_db
        )

        print("✅ Connected to database successfully!")

        try:
            # Execute the SQL commands
            print("🚀 Executing init.sql...")
            await conn.execute(sql_commands)
            print("✅ init.sql executed successfully!")

        except Exception as e:
            print(f"⚠️  Error executing SQL: {e}")
            print("\n💡 Tip: If tables already exist, you might see errors.")
            print("   This is normal if you're running init.sql multiple times.")

        finally:
            await conn.close()

    except Exception as e:
        print(f"❌ Cannot connect to database: {e}")
        print("\n🔧 Troubleshooting steps:")
        print("1. Check if PostgreSQL is running: sudo service postgresql status")
        print(f"2. Test connection: psql -h {db_config['host']} -U {db_config['user']} -d {target_db}")
        print(f"3. Verify database exists: psql -h {db_config['host']} -U {db_config['user']} -l")


if __name__ == "__main__":
    asyncio.run(run_init_sql())