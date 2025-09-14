import json, os
p='stage3/stage3_forgery_ratios.ipynb'
nb=json.load(open(p))
changed=False
for i,c in enumerate(nb['cells']):
    if c.get('cell_type')!='code':
        continue
    src=''.join(c.get('source',''))
    if 'def load_shift_data' in src and "_random_val.pkl')" in src and "_random_train.pkl')" in src:
        insert_after = "val_pkl = os.path.join(bags_dir, f'bags_arcface_margin_{margin}_{train_rtag}_random_val.pkl')"
        pref = (
            "\n    # Shift 전용 네임스페이스 파일이 있으면 우선 사용\n"
            "    train_pkl_shift = os.path.join(bags_dir, f'bags_arcface_margin_{margin}_{train_rtag}_random_train_shiftbase.pkl')\n"
            "    val_pkl_shift   = os.path.join(bags_dir, f'bags_arcface_margin_{margin}_{train_rtag}_random_val_shiftbase.pkl')\n"
            "    if os.path.exists(train_pkl_shift):\n"
            "        train_pkl = train_pkl_shift\n"
            "    if os.path.exists(val_pkl_shift):\n"
            "        val_pkl = val_pkl_shift\n\n"
        )
        new = src.replace(insert_after, insert_after + pref)
        if new!=src:
            c['source']=[new]
            changed=True
if changed:
    with open(p,'w') as f:
        json.dump(nb,f)
print('stage3 load_shift_data updated:', changed)
