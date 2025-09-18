import json
nb=json.load(open('stage2/stage2_forgery_ratios.ipynb'))
idx=6
c=nb['cells'][idx]
print(f"===== Cell {idx} =====")
print(''.join(c.get('source','')))
