import json
p='stage2/stage2_forgery_ratios.ipynb'
nb=json.load(open(p))
changed=False
# 1) Update shift base save tags in the [RUN] cell
for i,c in enumerate(nb['cells']):
    if c.get('cell_type')!='code':
        continue
    src=''.join(c.get('source',''))
    if 'Shift 모드: Train=30% 고정' in src and 'save_split(train_bags_30' in src:
        new=src
        new=new.replace("save_split(train_bags_30, train_labels_30, train_meta_30, 'train',", \
                        "save_split(train_bags_30, train_labels_30, train_meta_30, 'train_shiftbase',")
        new=new.replace("save_split(val_bags_30,   val_labels_30,   val_meta_30,   'val',", \
                        "save_split(val_bags_30,   val_labels_30,   val_meta_30,   'val_shiftbase',")
        if new!=src:
            c['source']=[new]
            changed=True
# 2) Update final summary lines listing shift train/val filenames
for i,c in enumerate(nb['cells']):
    if c.get('cell_type')!='code':
        continue
    src=''.join(c.get('source',''))
    target = "print(f\"    - bags_arcface_margin_{margin_value}_30p_baseline_train/val.pkl\")"
    if target in src:
        replacement = ("print(f\"    - bags_arcface_margin_{margin_value}_30p_baseline_train_shiftbase.pkl\")\n"
                       "print(f\"    - bags_arcface_margin_{margin_value}_30p_random_train_shiftbase.pkl\")\n"
                       "print(f\"    - bags_arcface_margin_{margin_value}_30p_baseline_val_shiftbase.pkl\")\n"
                       "print(f\"    - bags_arcface_margin_{margin_value}_30p_random_val_shiftbase.pkl\")")
        new = src.replace(target, replacement)
        if new!=src:
            c['source']=[new]
            changed=True
if changed:
    with open(p,'w') as f:
        json.dump(nb,f)
print('stage2 shift-namespace updated:', changed)
