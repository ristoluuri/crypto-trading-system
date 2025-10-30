import os, json

def load_symbol_quantities(base_dir: str, filename: str = "symbol_quantities10.json") -> dict:
    quantities_path = os.path.join(base_dir, "..", "..", filename)
    with open(quantities_path) as f:
        return json.load(f)
