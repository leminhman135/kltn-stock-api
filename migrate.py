"""Run database migrations on Railway."""
import subprocess
import sys

print("🔄 Running database migrations...")
result = subprocess.run(["alembic", "upgrade", "head"], capture_output=True, text=True)

print(result.stdout)
if result.stderr:
    print(result.stderr, file=sys.stderr)

if result.returncode == 0:
    print("✅ Migrations completed successfully!")
else:
    print("❌ Migrations failed!")
    sys.exit(1)
