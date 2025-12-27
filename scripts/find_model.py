import os

def find_model(start_path):
    print(f"Scanning {start_path}...")
    for root, dirs, files in os.walk(start_path):
        lower_root = root.lower()
        # Loose matching for "llama" and "8b"
        if "llama" in lower_root and "8b" in lower_root:
            if "config.json" in files:
                return root
    return None

search_paths = [
    "C:/Users/valno", # Trying the user root again but will handle permission errors by try-except in os.walk if needed (python handles it gracefully usually)
    "C:/Users/valno/models",
    "C:/Users/valno/Dev",
    "C:/models",
]

found_path = None
for path in search_paths:
    if os.path.exists(path):
        if path == "C:/Users/valno":
             # Restrict depth or ignore errors to avoid long scan
             pass 
        found_path = find_model(path)
        if found_path:
            break

if found_path:
    print(f"FOUND_MODEL_PATH: {found_path}")
else:
    print("Model still not found. Please provide path.")
