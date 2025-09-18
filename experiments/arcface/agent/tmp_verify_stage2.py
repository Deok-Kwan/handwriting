import json
nb=json.load(open('stage2/stage2_forgery_ratios.ipynb'))
for i,c in enumerate(nb['cells']):
    if c['cell_type']=='code' and 'RATIOS' in ''.join(c.get('source','')):
        print('Cell',i)
        print(''.join(c.get('source','')))
    if c['cell_type']=='code' and 'sample_ratios' in ''.join(c.get('source','')):
        print('Cell',i)
        print(''.join(c.get('source','')))
    if c['cell_type']=='markdown':
        src=''.join(c.get('source',''))
        if 'Stage 2' in src:
            print('Markdown updated? contains 50%:', '50%' in src)
