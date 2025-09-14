import json
nb=json.load(open('stage3/stage3_forgery_ratios.ipynb'))
print('cells:', len(nb.get('cells', [])))
for i,c in enumerate(nb.get('cells', [])):
    t=c.get('cell_type')
    src=''.join(c.get('source',''))
    first=' | '.join(src.strip().splitlines()[:3]) if src else ''
    print(f'[{i:02d}] {t}: {first[:240]}')
