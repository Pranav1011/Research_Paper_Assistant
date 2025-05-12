# tests/validate_mineru_output.py

import json
from pathlib import Path

print("🧪 Validating MinerU _model.json files...\n")

files = list(Path("data/mineru_parsed").glob("**/*_model.json"))
print(f"📂 Found {len(files)} files\n")

for f in files:
    try:
        with open(f, "r") as fp:
            data = json.load(fp)

        if not isinstance(data, list):
            raise TypeError("Expected a list at the top level")

        sample = data[0] if data else {}
        print(f"✅ {f.name} OK — {len(data)} blocks, first category: {sample.get('category_id', 'N/A')}")

    except Exception as e:
        print(f"❌ Error in {f.name}: {e}")