"""Japanese edition of the 2026-06-25 progress deck.

Reuses the layout helpers, colours, coordinates, generated figures and curated
media from ``build_report.py`` (the English generator) and only swaps the text to
Japanese. A final pass sets every run to a Japanese-capable font (Yu Gothic).

Figures keep their original English axis labels (they are data plots / schematics);
only the slide text is localized.

Run with the project venv (after build_report.py has populated figures + media):
    .venv\\Scripts\\python.exe report\\20260625\\build_report_ja.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_report as en  # noqa: E402  (helpers, colours, paths, media)

from pptx import Presentation  # noqa: E402
from pptx.enum.shapes import MSO_SHAPE  # noqa: E402
from pptx.oxml.ns import qn  # noqa: E402
from pptx.util import Inches, Pt  # noqa: E402
from pptx.dml.color import RGBColor  # noqa: E402

JP_FONT = "Yu Gothic"
PLOTS, IMAGES, VIDEOS, POSTERS = en.PLOTS, en.IMAGES, en.VIDEOS, en.POSTERS
NAVY, ACCENT, SUCCESS, WARN = en.NAVY, en.ACCENT, en.SUCCESS, en.WARN
DARK, GREY, WHITE = en.DARK, en.GREY, en.WHITE
GREEN_HDR, AMBER_HDR = en.GREEN_HDR, en.AMBER_HDR


def _set_ea(rPr, name):
    ea = rPr.find(qn("a:ea"))
    if ea is None:
        ea = rPr.makeelement(qn("a:ea"), {"typeface": name})
        latin = rPr.find(qn("a:latin"))
        latin.addnext(ea) if latin is not None else rPr.append(ea)
    else:
        ea.set("typeface", name)


def _font_runs(text_frame, name):
    for p in text_frame.paragraphs:
        for r in p.runs:
            r.font.name = name
            _set_ea(r._r.get_or_add_rPr(), name)


def apply_jp_font(prs, name=JP_FONT):
    for slide in prs.slides:
        for shp in slide.shapes:
            if shp.has_text_frame:
                _font_runs(shp.text_frame, name)
            if shp.has_table:
                for row in shp.table.rows:
                    for cell in row.cells:
                        _font_runs(cell.text_frame, name)


def build():
    prs = Presentation()
    prs.slide_width = Inches(en.SW)
    prs.slide_height = Inches(en.SH)
    blank, title_bar = en.blank, en.title_bar
    bul, img, vid, tbl = en.add_bullets, en.add_image_fit, en.add_video, en.add_table

    # ---- 1. Title
    s = blank(prs)
    band = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    band.fill.solid(); band.fill.fore_color.rgb = NAVY; band.line.fill.background(); band.shadow.inherit = False
    strip = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(4.7), prs.slide_width, Inches(0.08))
    strip.fill.solid(); strip.fill.fore_color.rgb = ACCENT; strip.line.fill.background(); strip.shadow.inherit = False
    tb = s.shapes.add_textbox(Inches(0.9), Inches(1.9), Inches(11.6), Inches(2.8)); tf = tb.text_frame; tf.word_wrap = True
    r = tf.paragraphs[0].add_run(); r.text = "2匹のオスは区別できる"
    r.font.size = Pt(42); r.font.bold = True; r.font.color.rgb = WHITE
    r2 = tf.add_paragraph().add_run(); r2.text = "完全な3個体識別トラック — そしてそれが可能にするオス別の配偶者選択"
    r2.font.size = Pt(20); r2.font.color.rgb = RGBColor(0x9F, 0xC5, 0xE8)
    tb2 = s.shapes.add_textbox(Inches(0.9), Inches(5.0), Inches(11.6), Inches(1.7)); tf2 = tb2.text_frame; tf2.word_wrap = True
    for i, (txt, sz, col) in enumerate([
        ("デュアルライン・トラッキング（幾何 ∥ 外見） ・ OIST saion HPC上のDetectron2 ・ K_1_2（メス1 + オス2）", 14, RGBColor(0xD7, 0xE3, 0xF0)),
        ("2026-06-22以降の更新：「オスは区別不可」という結論を覆した — 原因はラベルノイズで、外見の弱さではなかった。", 14, RGBColor(0xB9, 0xCD, 0xE0)),
        ("2026-06-25", 14, RGBColor(0x86, 0xA8, 0xC8)),
    ]):
        pp = tf2.paragraphs[0] if i == 0 else tf2.add_paragraph()
        rr = pp.add_run(); rr.text = txt; rr.font.size = Pt(sz); rr.font.color.rgb = col

    # ---- 2. Recap
    s = blank(prs); title_bar(s, prs, "前回（2026-06-22）の到達点")
    bul(s, [
        ("前回までに確立していた3点：", 0, NAVY, True),
        ("✓ キャリブレーション検証済（0.90 mm）。SLEAP round-5を実運用。", 1, SUCCESS, False),
        ("✓ 3D幾何は良好 — 348,600フレームで速度スパイク0、ID紐付け95–96%。", 1, SUCCESS, False),
        ("✓ メスを外見から確実に識別（P 0.87、マージン0.79）し、性別付きCSVに組込み。", 1, SUCCESS, False),
        ("未解決の問題 — そして当時の結論：", 0, AMBER_HDR, True),
        ("✗「2匹のオスは外見では区別できない」", 1, WARN, True),
        ("交絡除去テストは決定的に見えた：マスク済CNN 0.235 vs 位置ベースライン0.303 — どちらも偶然以下。", 1, DARK, False),
        ("→ オスは外見ではなく3D幾何＋運動リゾルバで解く方針だった。", 1, DARK, False),
        ("本報告：その結論は誤りだった。何が変わったかを示す。", 0, ACCENT, True),
    ], 0.5, 1.35, 7.0, 5.6, size=15)
    img(s, IMAGES / "3d_multiview_composite_still.png", 7.7, 1.5, 5.3, 4.6,
        caption="06-22時点：クリーンな3D＋マルチビュー、メスは固定、オスはまだダイアド（対）。")

    # ---- 3. The reversal
    s = blank(prs); title_bar(s, prs, "逆転 — 原因はウィンドウ間のラベルノイズ、外見の弱さではない", color=GREEN_HDR)
    bul(s, [
        ("以前のオステストを壊していた真因：", 0, NAVY, True),
        ("外見クロップは問題なかった — 学習ラベルがウィンドウ間で入れ替わっていた。", 1, DARK, True),
        ("3Dのglobal-idはウィンドウ間で置換される → プールすると識別ラベルが混ざる。", 1, DARK, False),
        ("ウィンドウ内では信号は強い（3クラスARI 0.82、オス対オス0.67）。プールで崩壊（0.24 / 0.11）。", 1, DARK, False),
        ("対策＝プール前にウィンドウ毎にラベルを整合（window_align）してから学習。", 0, ACCENT, True),
        ("スワップ補正後の結果：", 0, GREEN_HDR, True),
        ("✓ メス対オス LOSO 0.83/0.75 → 1.00/1.00；全体メスAUC 0.982。", 1, SUCCESS, True),
        ("✓ オス対オス：偶然 → balacc 0.936、AUC 0.994（位置のみベースライン0.477＝偶然）。", 1, SUCCESS, True),
        ("同じ修正で両者解決：オスは初めから区別可能だった。", 0, DARK, True),
    ], 0.5, 1.35, 6.7, 5.6, size=14)
    img(s, PLOTS / "fig_reversal_auc.png", 7.2, 1.5, 5.9, 5.1,
        caption="スワップ補正前後 — オス・メスの識別はともに偶然ラインを決定的に上回る。")

    # ---- 4. The dual-line method
    s = blank(prs); title_bar(s, prs, "手法 — 独立した2つのライン、シーンで融合")
    img(s, PLOTS / "fig_dual_line.png", 0.45, 1.3, 12.4, 3.6)
    bul(s, [
        ("ライン A — 幾何：キャリブレーションモデルで3D再投影＋速度/ギャップ考慮のハンガリアンID連結。広い場所で強いが、交差ではIDが滑らかに入れ替わる（速度スパイク指標に出ない）。", 0, ACCENT, False),
        ("ライン B — 外見：Detectron2マスク → 33._square_crop → フレーム毎の F / M1 / M2 分類器（＋上面モデル）。幾何が失敗する場所、接触・交差でこそ強い。", 0, SUCCESS, False),
        ("設計上相補的：ハードコアイベントの44.5%が3Dで不確実（Phase-0ゲート）— 2つのラインは別々のフレームで失敗する。", 0, NAVY, True),
        ("重要点：クロップは必ず33._square_crop（bbox→リサイズ、非マスク）— この一択でAUC 0.982 対 約0.5。", 0, WARN, True),
    ], 0.5, 5.15, 12.4, 2.2, size=12.5, gap=5)

    # ---- 5. The full 3-fish track
    s = blank(prs); title_bar(s, prs, "成果 — K_1_2の完全な3個体識別トラック", color=GREEN_HDR)
    img(s, PLOTS / "fig_track_quality.png", 0.45, 1.25, 12.4, 3.55,
        caption="トラック全体の識別内訳（左）と個体別の高信頼率（右）。")
    bul(s, [
        ("F + M1 + M2 を全フレーム、欠損なし：", 0, NAVY, True),
        ("✓ 348,600フレーム。各フレームちょうど 1 F + 1 M1 + 1 M2（違反0）。", 1, SUCCESS, True),
        ("✓ 高信頼98.6% ・ 補間1.3% ・ 曖昧0.06%。", 1, SUCCESS, True),
        ("✓ 個体別高信頼：F 99.0% ・ M1 98.4% ・ M2 98.4%。", 1, SUCCESS, True),
        ("0.06%の曖昧コア＝メスもオスも同時に全て融合する646魚フレーム。", 0, DARK, False),
        ("成果物：round5_260611_3d_individual_id_interp.csv （列 id_final / id_source / id_conf — 解析可能）。", 0, ACCENT, True),
    ], 0.5, 5.2, 12.4, 2.2, size=12.5, gap=2)

    # ---- 6. How it's built
    s = blank(prs); title_bar(s, prs, "構築の流れ — saion展開 → フレーム毎 → 融合 → 補間")
    bul(s, [
        ("1 ・ saion Detectron2 フルレングス展開", 0, ACCENT, True),
        ("3方向 × 348,600フレーム → /bucket上にマスク（24チャンク、約35 GB）。", 1, DARK, False),
        ("2 ・ フレーム毎の外見ID", 0, ACCENT, True),
        ("メスモデル（window_align）が3つの幾何スロット全てでメスを固定（430スワップを修正）。", 1, DARK, False),
        ("オスモデルがM1/M2を固定 — 全体で一貫（独立埋め込み器：ARI 0.83）。", 1, DARK, False),
        ("3 ・ 上面融合（人手ラベルなし）", 0, ACCENT, True),
        ("側面トラックから自動ラベル付けした上面モデルが側面低信頼区間を補完（コアの45.3%を回復）。", 1, DARK, False),
        ("4 ・ 残差を補間", 0, ACCENT, True),
        ("低信頼区間の86.5%は橋渡し可能（確実）。真のスワップは4.9%のみで、交差点で運動的に特定。", 1, DARK, False),
        ("オス信頼度の階段：側面78.5% → ＋上面88.2% → ＋補間98.4%。", 0, SUCCESS, True),
    ], 0.5, 1.35, 7.6, 5.7, size=13.5, gap=2)
    vid(s, VIDEOS / "3d_multiview_composite_clip.mp4", POSTERS / "3d_multiview_composite_clip.png",
        8.4, 1.55, 4.6, 3.6, caption="▶ 3D＋マルチビュー合成（埋込）。")
    img(s, POSTERS / "K_1_2_crossing_montage.png", 8.4, 5.55, 4.6, 1.55,
        caption="▶ videos/K_1_2_crossing_montage.mp4 — 融合が勝つべき交差。")

    # ---- 7. Validation
    s = blank(prs); title_bar(s, prs, "検証 — 妊娠腹部の形態 ＋ 埋め込み器の全体一貫性", color=GREEN_HDR)
    img(s, PLOTS / "gt_montage.png", 0.4, 1.3, 4.7, 5.9, caption="判定された F | M1 | M2 を時系列で（産卵含む）。")
    bul(s, [
        ("メス判定の外部検証（妊娠腹部）：", 0, NAVY, True),
        ("✓ 判定メスはどの行でも体高が高く丸みのある個体 —", 1, SUCCESS, True),
        ("しかも各行で異なる幾何スロットにいる（ウィンドウ間スワップ）、", 1, DARK, False),
        ("それでも外見は常に妊娠個体を選ぶ → スロットではなく生物学を追跡。", 1, DARK, False),
        ("これが0.99の内部メス検証を形態学的に裏付ける。", 1, GREY, False),
        ("オス判定の検証（M1 ≠ M2）：", 0, NAVY, True),
        ("オスは同サイズの細身個体 — 目視では区別不可（想定通り）。", 1, DARK, False),
        ("✓ 独立埋め込み器が全時系列で2つの安定した個体を確認：", 1, SUCCESS, True),
        ("2-means ARI 0.83、silhouette 0.81、セグメント間転移0.81–0.99 — 全体反転なし。", 1, DARK, False),
        ("つまりモデル＋埋め込み器がM1≠M2を、妊娠腹部がメスを担保。", 0, ACCENT, True),
    ], 5.4, 1.4, 7.6, 5.7, size=13.5, gap=3)

    # ---- 8. The payoff
    s = blank(prs); title_bar(s, prs, "成果の核心 — オス別の配偶者選択（K_1_2）", color=GREEN_HDR)
    img(s, PLOTS / "mate_choice.png", 7.05, 1.3, 6.0, 5.9,
        caption="上：メス–オス距離の推移。中：求愛範囲内の累積時間。下：選好指数。")
    bul(s, [
        ("研究課題 — ついに個体単位で回答可能：", 0, NAVY, True),
        ("✓ メスは外見M1を選好。", 0, SUCCESS, True),
        ("M1から30 mm以内287.3秒 vs M2 180.7秒（指数0.61）；", 1, DARK, False),
        ("接近遭遇の67.6%でM1により近い；全体でもM1に近いのが69.3%。", 1, DARK, False),
        ("✓ 選好は接触に向かって鋭くなる：", 0, SUCCESS, True),
        ("1.59倍（30 mm）→ 1.82倍（20）→ 1.81倍（12）→ 1.89倍（8 mm）。", 1, DARK, False),
        ("✓ M1の求愛が多い — 275回 / 271.3秒 vs 210回 / 162.2秒。", 0, SUCCESS, False),
        ("✓ 産卵（結果）＝M1：フレーム135,888（818.6秒）、", 0, SUCCESS, True),
        ("中央 d(F,M1) 7.4 mm vs d(F,M2) 11.5 mm — 接触。", 1, DARK, False),
        ("従来は不可能：BORISはダイアドしか採点できなかった（オス区別不可）。", 0, ACCENT, True),
    ], 0.5, 1.35, 6.6, 5.8, size=13.5, gap=3)

    # ---- 9. Temporal dynamics
    s = blank(prs); title_bar(s, prs, "時間的ダイナミクス — 選好は安定、そして求愛が鍵")
    bul(s, [
        ("選好は次第に強まるのか、後で切り替わるのか？", 0, NAVY, True),
        ("✓ M1へ安定 — ビン毎指数 前半0.65 → 後半0.66（傾き≈0）。", 1, SUCCESS, True),
        ("✓ 早期の選好が産卵結果を予測 — 最初からM1を選好。", 1, SUCCESS, True),
        ("（後からの転向ではない；選ばれたオスがほぼ一貫して優勢。）", 1, GREY, False),
        ("求愛 vs 攻撃 — 選択を実際に追うのはどちらか：", 0, NAVY, True),
        ("✓ 時間で求愛が攻撃を24.5倍上回る：", 1, SUCCESS, True),
        ("求愛485回 / 433.5秒  vs  攻撃171回 / 17.7秒。", 1, DARK, False),
        ("攻撃はほぼ対称（M1が速い始動側 97/171、57%）。", 1, DARK, False),
        ("→ オス間攻撃ではなく求愛強度がメスの選択に一致。", 0, ACCENT, True),
    ], 0.5, 1.35, 7.3, 5.0, size=14)
    rows = [
        ["", "M1（選択）", "M2"],
        ["求愛時間", "271.3秒", "162.2秒"],
        ["求愛回数", "275", "210"],
        ["30 mm以内", "287.3秒", "180.7秒"],
        ["選好指数(前→後)", "0.65 → 0.66", "安定"],
    ]
    tbl(s, rows, 7.9, 1.7, 5.0, col_w=[2.2, 1.6, 1.2], fs=12.5)
    bul(s, [
        ("求愛 : 攻撃 = 433.5秒 : 17.7秒 = 24.5倍", 0, GREEN_HDR, True),
    ], 7.9, 4.4, 5.0, 0.6, size=13)

    # ---- 10. BORIS cross-check
    s = blank(prs); title_bar(s, prs, "相互検証 — 手動BORISコーディングがオス別の非対称性で一致", color=GREEN_HDR)
    bul(s, [
        ("デュアルライントラックにより手動エソグラムを個体単位で読めるように：", 0, NAVY, True),
        ("BORISのオス→メス求愛：m1tf 41.8秒 vs m2tf 103.6秒（「m2」へ2.5倍）。", 1, DARK, False),
        ("→ BORISの「m2」＝外見の「M1」＝選ばれた相手。", 0, SUCCESS, True),
        ("2つの独立した測定が同じオスを強い求愛者／選ばれた相手と指す。", 1, DARK, True),
        ("オス別の非対称性（片方のオスが遥かに多く求愛）は外見でも手動コーディングでも再現。", 1, DARK, False),
        ("大きさの差について（なぜ想定内か）：", 0, NAVY, True),
        ("BORISは明示的ディスプレイを採点（2.5倍）。近接連合は緩やか（1.6倍）—", 1, DARK, False),
        ("だが接触域（8 mm以内で1.9倍）では両者は収束し、整合する。", 1, DARK, False),
        ("注意：BORISのm1/m2は注釈者の任意ラベル — 一致するのは方向であり、名前自体ではない；", 0, GREY, False),
        ("非対称性を特定の全体一貫した個体に割り当てるのは外見トラック。", 1, GREY, False),
    ], 0.6, 1.4, 12.2, 5.6, size=14.5)

    # ---- 11. N=3 extension
    s = blank(prs); title_bar(s, prs, "N = 3 への拡張 — K_1_1 + K_1_3（進行中）", color=NAVY)
    rows = [
        ["ライン B 要素", "K_1_1", "K_1_3", "K_1_2（参照）"],
        ["検出器（D2→マスク）", "✓ 転移", "✓ 転移", "✓ フル"],
        ["メスアンカー P(F)/マージン", "0.92 / 0.80", "0.82 / 0.64", "0.93"],
        ["オス対オス AUC(LOSO)", "0.948", "0.990", "0.994"],
        ["オス balacc/位置基準", "0.87 / 0.62", "0.97 / 0.57", "0.94 / 0.48"],
    ]
    tbl(s, rows, 0.5, 1.5, 8.1, col_w=[3.0, 1.7, 1.7, 1.7], fs=12)
    bul(s, [
        ("✓ C0ゲート通過 — ライン Bの3要素が両新実験に転移、", 0, SUCCESS, True),
        ("新規推論ゼロで（既存のサンプルウィンドウ成果物＋数分のローカルGPU）。", 1, DARK, False),
        ("メスは両方で非曖昧かつウィンドウ間一貫；オス対オスも決定的に基準を超える。", 0, DARK, False),
        ("状況：saionフルレングスDetectron2展開を2026-06-25に投入", 0, ACCENT, True),
        ("（2実験 × 3方向 × 8チャンク）— マスクを計算中。", 1, DARK, False),
        ("マスク到着後：実験毎のライン B → 3個体トラック → analyze_mate_choice → N=3の結果。", 0, NAVY, True),
        ("注意：C0はクリーンクロップの上限；接触コアはK_1_2同様に下流で処理（上面融合＋補間）。", 0, GREY, False),
    ], 0.5, 4.0, 12.4, 3.2, size=13)

    # ---- 12. Production & reproducibility
    s = blank(prs); title_bar(s, prs, "実運用と再現性")
    bul(s, [
        ("使い捨ての解析ではない — デュアルライン手法をパイプラインに組込み：", 0, NAVY, True),
        ("✓ per_frame_resolve.py — 解決/融合/補間の単一の真実源（単体テスト済）。", 0, SUCCESS, False),
        ("✓ mate_choice_core.py — 共有指標コア；2つの解析プレースホルダがこれで稼働。", 0, SUCCESS, False),
        ("✓ 40_female_anchor_pipelineのステージ5（--per-frame-resolve）が3個体トラックを一気通貫で出力。", 0, SUCCESS, False),
        ("✓ resolve_hardcore_events --appearance-id-csv が924件の難イベントを裁定（一致415、フラグ45）。", 0, SUCCESS, False),
        ("✓ 検証済CSVをバイト単位で再現（md5不変）— 抽出は忠実。", 0, SUCCESS, False),
        ("✓ 269テスト通過；コミット＋プッシュ → PR #2（oist/medaka-mating-pipeline#2）。", 0, SUCCESS, True),
        ("各ステージがCSV/JSONをディスク出力 → 各工程は独立に再実行・検査可能。", 0, GREY, False),
        ("本デッキ自体：report/20260625/build_report.py（＋_ja）が図・スライド・プレビューを再生成。", 0, ACCENT, True),
    ], 0.6, 1.4, 12.2, 5.6, size=14.5)

    # ---- 13. Summary & next steps
    s = blank(prs); title_bar(s, prs, "まとめと次のステップ")
    bul(s, [
        ("2026-06-22以降の変化：", 0, NAVY, True),
        ("✓「2オス区別不可」の結論を覆した — 原因はラベルノイズ（スワップ補正後オスAUC 0.994）。", 0, SUCCESS, False),
        ("✓ デュアルライン（幾何 ∥ 外見）で完全な3個体トラック — 高信頼98.6%、欠損なし。", 0, SUCCESS, False),
        ("✓ オス別配偶者選択（K_1_2）：メスはM1を選択 — 求愛＋産卵 — 最初から安定した選好。", 0, SUCCESS, False),
        ("✓ 検証済（妊娠腹部＋埋め込み器の全体一貫性）、BORISが方向で裏付け。", 0, SUCCESS, False),
        ("✓ 実運用化＋プッシュ（PR #2）；時間で求愛が攻撃を24.5倍上回る。", 0, SUCCESS, False),
        ("次：", 0, NAVY, True),
        ("→ N=3を完了：saionマスク到着後にK_1_1 / K_1_3の実験毎ライン B＋配偶者選択。", 1, ACCENT, False),
        ("→ 妊娠腹部判定を目視確認（生物学的承認）、可能なら産卵時の父性。", 1, ACCENT, False),
        ("→ 近接ラベルでround-6 SLEAPを学習し、融合コアを根本から縮小。", 1, ACCENT, False),
        ("→ 研究室の残る配偶者選択指標（頭部同期、クイックサークル半径）を3個体トラックに追加。", 1, ACCENT, False),
    ], 0.6, 1.4, 12.2, 5.7, size=14)

    apply_jp_font(prs, JP_FONT)
    out = HERE / "medaka_progress_report_20260625_JA.pptx"
    prs.save(str(out))
    return out


if __name__ == "__main__":
    out = build()
    print(f"Japanese deck written: {out}  ({out.stat().st_size/1e6:.1f} MB)")
