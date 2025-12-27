import os

def check_model():
    base = r"C:\Users\valno\.cache\huggingface\hub\models--meta-llama--Llama-3.1-8B-Instruct\snapshots"
    if os.path.exists(base):
        snapshots = os.listdir(base)
        if snapshots:
            full_path = os.path.join(base, snapshots[0]) # Take the first one
            print(f"MODEL_PATH: {full_path}")
            return full_path
    return None

def list_data_files():
    base = "data/raw/webnlg"
    print("Listing sample data files:")
    count = 0
    for root, dirs, files in os.walk(base):
        for f in files:
            if f.endswith(".xml"):
                print(os.path.join(root, f))
                count += 1
                if count >= 3: return # Just need a few

if __name__ == "__main__":
    check_model()
    list_data_files()
