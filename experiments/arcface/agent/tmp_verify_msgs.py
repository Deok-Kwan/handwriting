import json
nb=json.load(open('stage2/stage2_forgery_ratios.ipynb'))
for i,c in enumerate(nb['cells']):
    if c['cell_type']=='code' and 'Stage 3에서 다음과 같이 활용' in ''.join(c.get('source','')):
        print('Updated cell', i)
        print('\n'.join([line for line in ''.join(c.get('source','')).splitlines() if 'Matched 모드' in line or 'Shift 모드' in line]))
