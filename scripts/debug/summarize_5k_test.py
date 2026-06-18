"""One-off: summarise detection quality of the round5 5k-frame test outputs."""
from pathlib import Path

from scripts.sleap.compare_rounds import summarise_slp

BASE = Path(r"Z:/StephensU/medaka/sleap_output/round5_5k_test")
EXPS = ["20251225_K_1_1_FCMC", "20251225_K_1_2_FCMC", "20251225_K_1_3_FCMC"]
VIEWS = ["top_cam", "side1_cam", "side2_cam"]

hdr = (f"{'exp/view':<30}{'n':>6}{'=3%':>7}{'>=4%':>7}{'<=1%':>7}"
       f"{'inst_sc':>9}{'mt_p50':>8}{'mt<10px%':>10}")
print(hdr)
print("-" * len(hdr))
for e in EXPS:
    for v in VIEWS:
        p = BASE / e / "round5_260611" / f"{v}.slp"
        if not p.exists():
            print(f"{e[9:] + '/' + v:<30} MISSING")
            continue
        s = summarise_slp(p)
        h = s["inst_count_hist"]
        tot = max(1, s["n_frames_total"])
        p3 = 100 * h.get(3, 0) / tot
        pge4 = 100 * sum(h.get(k, 0) for k in (4, 5, 6)) / tot
        ple1 = 100 * sum(h.get(k, 0) for k in (0, 1)) / tot
        mt = s["mouth_tail"]
        print(f"{e[9:] + '/' + v:<30}{s['n_frames_total']:>6}{p3:>6.0f}%"
              f"{pge4:>6.0f}%{ple1:>6.0f}%{s['mean_inst_score']:>9.3f}"
              f"{mt['p50']:>8.1f}{mt['frac_lt_10'] * 100:>9.1f}%")
    print()
