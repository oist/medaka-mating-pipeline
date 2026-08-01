import numpy as np
import pandas as pd

TRACKS = {
    'NEW': 'Z:/StephensU/medaka/runs/udmt_validation/recon_nofill/20251225_K_1_2_FCMC/round5_260611_recovered_3d_individual_id_interp_debounced.csv',
    'OLD': 'runs/dual_line/lineB_fulltimeline/step2/round5_260611_3d_individual_id_interp_debounced.csv',
}

TELEPORT_MM = 60.0
SPAWN_FRAMES = [135888, 135883]
SPAWN_RANGE = range(135838, 135939)

rows = []

def dist(a, b):
    return float(np.sqrt(((a - b) ** 2).sum()))

for name, path in TRACKS.items():
    df = pd.read_csv(path)
    total_frames = df['frame'].nunique()

    # ---- 1. per-frame invariant: exactly 1 F + 1 M1 + 1 M2 ----
    g = df.groupby('frame')['id_final'].apply(lambda s: tuple(sorted(s.value_counts().reindex(['F','M1','M2']).fillna(0).astype(int))))
    # need exactly (1,1,1)
    counts = df.groupby(['frame','id_final']).size().unstack(fill_value=0)
    for c in ['F','M1','M2']:
        if c not in counts.columns:
            counts[c] = 0
    ok_mask = (counts['F'] == 1) & (counts['M1'] == 1) & (counts['M2'] == 1)
    n_ok = int(ok_mask.sum())
    n_viol = total_frames - n_ok
    # break down violations
    n_missing = int(((counts[['F','M1','M2']] == 0).any(axis=1)).sum())
    n_dup = int(((counts[['F','M1','M2']] > 1).any(axis=1)).sum())

    # ---- 2. id_conf coverage per fish ----
    conf_cov = {}
    for fish in ['F','M1','M2']:
        sub = df[df['id_final'] == fish]
        n = len(sub)
        vc = sub['id_conf'].value_counts(normalize=True)
        conf_cov[fish] = {k: float(vc.get(k, 0.0)) for k in ['high','interp','ambiguous']}
        conf_cov[fish]['n'] = n

    # ---- 3. SPAWN validation ----
    def pos_at(frame, fish):
        sub = df[(df['frame'] == frame) & (df['id_final'] == fish)]
        if len(sub) == 0:
            return None
        r = sub.iloc[0]
        v = np.array([r['X'], r['Y'], r['Z']], dtype=float)
        if np.isnan(v).any():
            return None
        return v

    spawn = {}
    for fr in SPAWN_FRAMES:
        Fp = pos_at(fr, 'F'); M1p = pos_at(fr, 'M1'); M2p = pos_at(fr, 'M2')
        dM1 = dist(Fp, M1p) if (Fp is not None and M1p is not None) else np.nan
        dM2 = dist(Fp, M2p) if (Fp is not None and M2p is not None) else np.nan
        nearest = 'M1' if (not np.isnan(dM1) and (np.isnan(dM2) or dM1 <= dM2)) else 'M2'
        spawn[fr] = {'dM1': dM1, 'dM2': dM2, 'nearest': nearest, 'min': float(np.nanmin([dM1, dM2]))}

    # min over range
    range_min = np.inf
    range_argmin = None
    for fr in SPAWN_RANGE:
        Fp = pos_at(fr, 'F'); M1p = pos_at(fr, 'M1'); M2p = pos_at(fr, 'M2')
        for mp, mn in [(M1p,'M1'),(M2p,'M2')]:
            if Fp is not None and mp is not None:
                d = dist(Fp, mp)
                if d < range_min:
                    range_min = d; range_argmin = (fr, mn)
    spawn_pass = (range_min <= 12.0)

    # ---- 4. teleports: >60mm jump between consecutive frames per fish ----
    teleports = {}
    for fish in ['F','M1','M2']:
        sub = df[df['id_final'] == fish].sort_values('frame')
        # one row per frame expected; if dup take first
        sub = sub.drop_duplicates('frame', keep='first').set_index('frame')[['X','Y','Z']]
        sub = sub.reindex(range(int(df['frame'].min()), int(df['frame'].max())+1))
        d = np.sqrt((sub.diff()**2).sum(axis=1))
        # only count where both frames present and consecutive (diff index ==1 guaranteed by reindex)
        teleports[fish] = int((d > TELEPORT_MM).sum())

    # ---- 5. slot churn ----
    # F label changes geometry_slot across timeline
    slot_churn = {}
    for fish in ['F','M1','M2']:
        sub = df[df['id_final'] == fish].sort_values('frame').drop_duplicates('frame', keep='first')
        gs = sub['geometry_slot'].values
        changes = int((np.diff(gs) != 0).sum())
        slot_churn[fish] = changes
    # M1<->M2 direct exchanges: frames where M1 and M2 swap their geometry_slot vs previous frame
    m1 = df[df['id_final']=='M1'].sort_values('frame').drop_duplicates('frame',keep='first').set_index('frame')['geometry_slot']
    m2 = df[df['id_final']=='M2'].sort_values('frame').drop_duplicates('frame',keep='first').set_index('frame')['geometry_slot']
    common = m1.index.intersection(m2.index)
    m1c = m1.reindex(common); m2c = m2.reindex(common)
    pm1 = m1c.shift(1); pm2 = m2c.shift(1)
    mm_swap = int(((m1c == pm2) & (m2c == pm1) & (m1c != pm1)).sum())

    # ---- emit rows ----
    def add(metric, value):
        rows.append({'track': name, 'metric': metric, 'value': value})

    add('total_frames', total_frames)
    add('total_rows', len(df))
    add('inv_frames_OK_1F1M1_1M2', n_ok)
    add('inv_violations', n_viol)
    add('inv_viol_missing_fish', n_missing)
    add('inv_viol_duplicate_label', n_dup)
    add('inv_OK_fraction', round(n_ok/total_frames, 6))
    for fish in ['F','M1','M2']:
        add(f'idconf_{fish}_n', conf_cov[fish]['n'])
        add(f'idconf_{fish}_high_frac', round(conf_cov[fish]['high'],6))
        add(f'idconf_{fish}_interp_frac', round(conf_cov[fish]['interp'],6))
        add(f'idconf_{fish}_ambiguous_frac', round(conf_cov[fish]['ambiguous'],6))
    for fr in SPAWN_FRAMES:
        add(f'spawn_f{fr}_dF_M1_mm', round(spawn[fr]['dM1'],3))
        add(f'spawn_f{fr}_dF_M2_mm', round(spawn[fr]['dM2'],3))
        add(f'spawn_f{fr}_nearest', spawn[fr]['nearest'])
        add(f'spawn_f{fr}_min_mm', round(spawn[fr]['min'],3))
    add('spawn_range_min_mm', round(float(range_min),3))
    add('spawn_range_argmin_frame', range_argmin[0] if range_argmin else None)
    add('spawn_range_argmin_male', range_argmin[1] if range_argmin else None)
    add('spawn_PASS', bool(spawn_pass))
    add('teleport_F_gt60mm', teleports['F'])
    add('teleport_M1_gt60mm', teleports['M1'])
    add('teleport_M2_gt60mm', teleports['M2'])
    add('teleport_total', teleports['F']+teleports['M1']+teleports['M2'])
    add('slotchurn_F', slot_churn['F'])
    add('slotchurn_M1', slot_churn['M1'])
    add('slotchurn_M2', slot_churn['M2'])
    add('slotchurn_M1M2_direct_exchange', mm_swap)

    print(f'=== {name} done ===')

out = pd.DataFrame(rows)
out_path = 'report/20260630/k12_appearance/bakeoff/B1_harness_compare.csv'
out.to_csv(out_path, index=False)
# pivot for readability
piv = out.pivot(index='metric', columns='track', values='value')
piv = piv.reindex([r['metric'] for r in rows if r['track']=='NEW'])
print(piv.to_string())
