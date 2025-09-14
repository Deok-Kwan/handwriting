import json,re
p='stage2/stage2_forgery_ratios.ipynb'
nb=json.load(open(p))
changed=False
# Update markdown ratio mentions
for c in nb['cells']:
    if c.get('cell_type')=='markdown':
        src=''.join(c.get('source',''))
        new=src
        new=new.replace('5/10/20/30%','5/10/20/30/50%')
        new=new.replace('(5/10/20/30% Forgery)','(5/10/20/30/50% Forgery)')
        new=new.replace('5%, 10%, 20%, 30%','5%, 10%, 20%, 30%, 50%')
        if new!=src:
            c['source']=[new]
            changed=True
# Update RATIOS
pat = re.compile(r"RATIOS\s*=\s*\[[^\]]*\]")
for c in nb['cells']:
    if c.get('cell_type')=='code':
        src=''.join(c.get('source',''))
        new=src
        if 'RATIOS' in src:
            new = re.sub(pat, 'RATIOS = [0.05, 0.10, 0.20, 0.30, 0.50]', new)
        if 'sample_ratios' in src:
            new = new.replace('sample_ratios = [0.05, 0.30]', 'sample_ratios = [0.05, 0.50]')
        if new!=src:
            c['source']=[new]
            changed=True
if changed:
    with open(p,'w') as f:
        json.dump(nb,f)
print('stage2 updated:', changed)
