import json
p='stage2/stage2_forgery_ratios.ipynb'
nb=json.load(open(p))
changed=False
for c in nb['cells']:
    if c.get('cell_type')=='code':
        src=''.join(c.get('source',''))
        new=src
        new=new.replace('각 비율별 독립 학습 4회', '각 비율별 독립 학습 {len(RATIOS)}회')
        new=new.replace('5/10/20/30% 평가', '5/10/20/30/50% 평가')
        if new!=src:
            c['source']=[new]
            changed=True
if changed:
    with open(p,'w') as f:
        json.dump(nb,f)
print('stage2 messages updated:', changed)
