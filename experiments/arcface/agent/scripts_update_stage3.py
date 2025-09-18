import json,re
p='stage3/stage3_forgery_ratios.ipynb'
nb=json.load(open(p))
changed=False
# Update markdown list of ratios
for c in nb['cells']:
    if c.get('cell_type')=='markdown':
        src=''.join(c.get('source',''))
        if '위조 비율' in src:
            new=src.replace('(5%, 10%, 20%, 30%)','(5%, 10%, 20%, 30%, 50%)')
            if new!=src:
                c['source']=[new]
                changed=True
# Update defaults and RATIOS
def repl_ratios_list(src):
    # Replace occurrences of a ratio list in function defaults
    src2 = src
    src2 = src2.replace('ratios=[0.05, 0.10, 0.20, 0.30]', 'ratios=[0.05, 0.10, 0.20, 0.30, 0.50]')
    src2 = src2.replace('eval_ratios=[0.05, 0.10, 0.20, 0.30]', 'eval_ratios=[0.05, 0.10, 0.20, 0.30, 0.50]')
    src2 = src2.replace('RATIOS = [0.05, 0.10, 0.20, 0.30]', 'RATIOS = [0.05, 0.10, 0.20, 0.30, 0.50]')
    return src2
for c in nb['cells']:
    if c.get('cell_type')=='code':
        src=''.join(c.get('source',''))
        new=repl_ratios_list(src)
        if new!=src:
            c['source']=[new]
            changed=True
if changed:
    with open(p,'w') as f:
        json.dump(nb,f)
print('stage3 updated:', changed)
