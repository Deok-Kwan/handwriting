import json
nb=json.load(open('stage2/stage2_forgery_ratios.ipynb'))
print('Verifying Stage 2 changes:')
for i,c in enumerate(nb['cells']):
    if c.get('cell_type')!='code':
        continue
    s=''.join(c.get('source',''))
    if 'Shift 모드: Train=30% 고정' in s and 'save_split(train_bags_30' in s:
        print('RUN cell updated:', 'train_shiftbase' in s and 'val_shiftbase' in s)
    if '📊 Shift 모드' in s:
        print('Summary mentions shiftbase names:', 'train_shiftbase' in s and 'val_shiftbase' in s)
