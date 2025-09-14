import json
nb=json.load(open('stage2/stage2_forgery_ratios.ipynb'))
for idx in [4,5,6,7,8,9]:
    c=nb['cells'][idx]
    print(f"\n===== Cell {idx} ({c['cell_type']}) =====")
    if c['cell_type']=='code':
        print(''.join(c.get('source','')))
    else:
        print(''.join(c.get('source',''))[:500])
