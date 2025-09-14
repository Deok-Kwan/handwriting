import json
nb=json.load(open('stage3/stage3_forgery_ratios.ipynb'))
for idx,c in enumerate(nb['cells']):
    if c['cell_type']!='code':
        continue
    src=''.join(c.get('source',''))
    if any(k in src for k in ['RATIOS =', 'ratio_to_tag', 'load_bags', 'Matched 모드', 'Shift 모드', 'RATIOS=']):
        print(f"\n===== cell {idx} =====\n{src}")
