"""
Root wrapper for Phase 2 — Step 2.3: Create & Verify Sample Graph Script.
"""

import sys
import asyncio
from pathlib import Path

# Add backend directory to sys.path
root_dir = Path(__file__).resolve().parent.parent
backend_dir = root_dir / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from scripts.verify_sample_graph import main

if __name__ == "__main__":
    asyncio.run(main())
