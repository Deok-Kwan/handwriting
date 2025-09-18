import json, re, sys
from pathlib import Path

def print_model_cells(nb_path: str):
    nb = json.load(open(nb_path))
    print(f"\n=== {nb_path} ===")
    print('cells:', len(nb.get('cells', [])))
    for i, c in enumerate(nb.get('cells', [])):
        if c.get('cell_type') != 'code':
            continue
        src = ''.join(c.get('source', ''))
        if re.search(r'class\s+(TransformerMIL|TransMIL)|TransformerEncoder|MultiheadAttention|Positional|AttentionPooler', src):
            print(f"\n--- code cell {i} ---\n{src}")

if __name__ == '__main__':
    for p in sys.argv[1:]:
        print_model_cells(p)
