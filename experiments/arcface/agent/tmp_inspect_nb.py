import json, sys
p = 'stage2/stage2_forgery_ratios.ipynb'
nb = json.load(open(p))
print('nbformat:', nb.get('nbformat'), nb.get('nbformat_minor'))
print('cells:', len(nb.get('cells', [])))
for i, c in enumerate(nb.get('cells', [])):
    t = c.get('cell_type')
    src = ''.join(c.get('source', ''))
    first = ' | '.join(src.strip().splitlines()[:3])
    print(f'[{i:02d}] {t}: {first[:200]}')
