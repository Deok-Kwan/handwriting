import json
nb=json.load(open('stage3/stage3_forgery_ratios.ipynb'))
for i,c in enumerate(nb['cells']):
    src=''.join(c.get('source','')) if c['cell_type']=='code' else ''.join(c.get('source',''))
    if c['cell_type']=='markdown' and i==1:
        print('Markdown cell updated? 50% in text:', '50%' in src)
    if c['cell_type']=='code' and 'RATIOS =' in src:
        print('RATIOS cell', i, [line for line in src.splitlines() if 'RATIOS' in line][0])
    if c['cell_type']=='code' and 'load_forgery_data' in src:
        print('Defaults updated in load_forgery_data:', '0.50' in src)
    if c['cell_type']=='code' and 'load_shift_data' in src:
        print('Defaults updated in load_shift_data:', '0.50' in src)
