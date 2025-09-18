import json
nb=json.load(open('stage3/stage3_forgery_ratios.ipynb'))
for i,c in enumerate(nb['cells']):
    if c.get('cell_type')!='code':
        continue
    s=''.join(c.get('source',''))
    if 'def load_shift_data' in s:
        print('Found load_shift_data cell:', i)
        print('\n'.join([line for line in s.splitlines() if 'train_pkl' in line or 'val_pkl' in line]))
