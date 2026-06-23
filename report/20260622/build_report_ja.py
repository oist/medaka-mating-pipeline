"""Japanese edition of the 2026-06-22 progress deck.

Reuses the layout helpers, colours, coordinates and curated media from
``build_report.py`` (the English generator) and only swaps the text to Japanese.
A final pass sets every run to a Japanese-capable font (Yu Gothic).

Figures keep their original English axis labels (they are data plots / external QC
artifacts); only the slide text is localized.

Run with the project venv (after build_report.py has populated the media folders):
    .venv\\Scripts\\python.exe report\\20260622\\build_report_ja.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_report as en  # noqa: E402  (helpers, colours, paths, media)

from pptx import Presentation  # noqa: E402
from pptx.enum.shapes import MSO_SHAPE  # noqa: E402
from pptx.enum.text import PP_ALIGN  # noqa: E402
from pptx.oxml.ns import qn  # noqa: E402
from pptx.util import Inches, Pt  # noqa: E402
from pptx.dml.color import RGBColor  # noqa: E402

JP_FONT = "Yu Gothic"
PLOTS, IMAGES, VIDEOS, POSTERS = en.PLOTS, en.IMAGES, en.VIDEOS, en.POSTERS
BC = HERE / "behavior_comparison"
NAVY, ACCENT, SUCCESS, WARN = en.NAVY, en.ACCENT, en.SUCCESS, en.WARN
DARK, GREY, WHITE = en.DARK, en.GREY, en.WHITE


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
    tb = s.shapes.add_textbox(Inches(0.9), Inches(2.0), Inches(11.6), Inches(2.6)); tf = tb.text_frame; tf.word_wrap = True
    r = tf.paragraphs[0].add_run(); r.text = "メダカ3個体 交配行動トラッキング"
    r.font.size = Pt(40); r.font.bold = True; r.font.color.rgb = WHITE
    r2 = tf.add_paragraph().add_run(); r2.text = "マルチカメラ3次元パイプライン — 進捗報告"
    r2.font.size = Pt(22); r2.font.color.rgb = RGBColor(0x9F, 0xC5, 0xE8)
    tb2 = s.shapes.add_textbox(Inches(0.9), Inches(5.0), Inches(11.6), Inches(1.7)); tf2 = tb2.text_frame; tf2.word_wrap = True
    for i, (txt, sz, col) in enumerate([
        ("キャリブレーション ・ SLEAP姿勢推定 ・ 3次元再構成 ・ Detectron2 外見ID ・ メスアンカー", 15, RGBColor(0xD7, 0xE3, 0xF0)),
        ("対象データ：20251225 Kグループ（K_1_1 / K_1_2 / K_1_3）  —  メス1 + オス2、166 fps、20 cm水槽", 14, RGBColor(0xB9, 0xCD, 0xE0)),
        ("2026-06-22", 14, RGBColor(0x86, 0xA8, 0xC8)),
    ]):
        pp = tf2.paragraphs[0] if i == 0 else tf2.add_paragraph()
        rr = pp.add_run(); rr.text = txt; rr.font.size = Pt(sz); rr.font.color.rgb = col

    # ---- 2. Goal & pipeline
    s = blank(prs); title_bar(s, prs, "目的とパイプライン全体像")
    bul(s, [
        ("研究課題：メス1+オス2の系で、オスの攻撃 vs 求愛強度からメスの配偶者選択を定量化する。", 0, DARK, True),
        "工程：撮影 → キャリブレーション → SLEAP姿勢推定（HPC） → 2D/3Dトラッキング → 3次元再構成 → 行動解析。",
        "同期3カメラ（上面XY + 側面2方向）、166 fps、20 cm水槽。各工程の出力はCSV/JSON（再実行・検証可能）。",
        ("現在の到達点：", 0, ACCENT, True),
        ("✓ キャリブレーションモデル検証済み（平均誤差0.90 mm）。", 1, SUCCESS, False),
        ("✓ SLEAPはround-5。3次元の幾何は良好（348,600フレームで速度スパイク0）。", 1, SUCCESS, False),
        ("✓ メスを外見から識別可能 — アンカーをパイプラインに組込み済み。", 1, SUCCESS, False),
        ("→ 残る課題：ほぼ同一の2オスを交差時に区別すること（幾何・連続性の問題）。", 1, WARN, False),
    ], 0.5, 1.35, 6.6, 5.6, size=15)
    img(s, IMAGES / "3d_multiview_composite_still.png", 7.4, 1.5, 5.5, 4.7,
        caption="ライブ映像：上面+側面2カメラ（上）と再構成した3個体の3D軌跡（下）。")

    # ---- 3. Calibration
    s = blank(prs); title_bar(s, prs, "キャリブレーション — 学習型 6D→3D マッピング（屈折モデル不要）")
    bul(s, [
        ("手法：多項式回帰で6次元画素座標（3カメラ）→3次元世界座標を学習。古典的ピンホールモデルではない。", 0, DARK, True),
        "水・ガラスによる屈折を暗黙的に吸収 → 明示的な屈折モデルが不要。",
        "較正物体：190 mm立方体、色分けビーズ49個（7×7）+ 角8点、4姿勢で撮影。",
        ("実運用精度（ホールドアウト228点）：", 0, ACCENT, True),
        ("✓ 平均0.90 mm  ・  RMSE 1.00 mm  ・  最大2.15 mm", 1, SUCCESS, True),
        "体の重心（25–35 mm）や個体間距離には十分。<2 mmの微小付属肢・微細なZには不向き。",
        ("拡張：2回目撮影向けに剛体物体リファインメント（Kabschポーズ+再学習）を構築。誤差が悪化しない場合のみ採用。", 0, GREY, False),
    ], 0.5, 1.35, 6.5, 5.6, size=15)
    img(s, PLOTS / "calibration_error_3d.png", 7.2, 1.4, 5.8, 3.0, caption="立方体全体の3D誤差 — 一様に2 mm未満。")
    img(s, PLOTS / "calibration_reprojection_2d.png", 7.2, 4.75, 5.8, 2.3, caption="3カメラでの2D再投影チェック。")

    # ---- 4. SLEAP
    s = blank(prs); title_bar(s, prs, "SLEAP姿勢推定 — 反復再学習（round 3→4→5）")
    bul(s, [
        ("2ノード骨格（口→尾）。centroid + centered-instance モデルを上面・側面で。", 0, DARK, True),
        "round-5（2026-06-11）：上面 n=900 / 側面 n=1800 ラベルフレーム（round-4比 約+35%）。",
        ("重要な修正：centered-instance のクロップを 256→448 px に拡大（魚が見切れていた）。", 0, ACCENT, True),
        ("round-5 vs round-4（ホールドアウト）：", 0, DARK, True),
        ("✓ centered-instance mOKS：上面 0.66→0.74、側面 0.58→0.69", 1, SUCCESS, False),
        ("✓ 側面 PCK@5 0.78→0.90；側面の最悪値（p99）75→60 px", 1, SUCCESS, False),
        ("✓ centroid はほぼ完璧（mOKS 0.99–1.00）", 1, SUCCESS, False),
        ("✗ 1点後退：上面 p99 口/尾 45→56 px → 次ラウンドで上面の難例を追加", 1, WARN, False),
        ("フルレングス推論は OIST saion HPC（A100）：348,600フレーム/方向 × 3実験。", 0, GREY, False),
    ], 0.5, 1.35, 6.2, 5.6, size=14)
    img(s, PLOTS / "sleap_round4_vs_round5_eval.png", 6.9, 1.4, 6.1, 5.4,
        caption="ホールドアウト評価：round-5（青）が全般に精度向上。右下のp99=口/尾の入れ替わり指標。")

    # ---- 5. SLEAP failure mode
    s = blank(prs); title_bar(s, prs, "SLEAP — 近接時の失敗モードと進行中の対策")
    bul(s, [
        ("残る誤りは交差時に集中（広い場所では起きない）。", 0, DARK, True),
        ("メカニズム（レビューで特定）：近接時、2個体は別々の口を保つが尾を1本共有 —", 0, DARK, False),
        ("負け側の重心が隣個体側へ約半身分引っ張られ、3D位置が収束 → ID入れ替わり。", 1, WARN, False),
        ("共有尾デデュープ（実装・テスト済）：共有された尾をnullにし口を残す。各方向で約6–7%のフレーム —", 0, DARK, False),
        ("交差に集中（最も近接する40交差の40/40でウィンドウ内に共有尾フレーム）。", 1, DARK, False),
        ("round-6 進行中：近接420フレームの注釈シードを構築。注釈→ウォームスタート学習が本命の修正。", 0, ACCENT, True),
        ("これらの入れ替わりは滑らかで残差・速度スパイクに現れない → 交差を目視確認する必要。", 0, GREY, False),
    ], 0.5, 1.35, 7.4, 5.0, size=14.5)
    img(s, PLOTS / "detectron2_vs_sleap_top.png", 8.1, 1.5, 4.9, 2.5, caption="上面：Detectron2マスク（塗り）vs SLEAP骨格（黄）。")
    img(s, POSTERS / "K_1_2_crossing_montage.png", 8.1, 4.35, 4.9, 2.4, caption="▶ videos/K_1_2_crossing_montage.mp4 — 交差での確認。")

    # ---- 6. 3D quality
    s = blank(prs); title_bar(s, prs, "現在の3Dトラッキング品質 — 1225 Kグループ（フルレングス、round-5）")
    rows = [
        ["実験", "カバレッジ", "残差 中央/p90 (px²)", "frac>5000", "速度スパイク", "ID紐付け"],
        ["K_1_1", "87.4 %", "137 / 432", "0.00 %", "0.000", "96.4 %"],
        ["K_1_2", "87.2 %", "132 / 457", "0.00 %", "0.000", "96.1 %"],
        ["K_1_3", "85.1 %", "107 / 437", "0.00 %", "0.000", "95.0 %"],
    ]
    tbl(s, rows, 0.5, 1.4, 7.2, col_w=[1.1, 1.2, 1.8, 1.0, 1.2, 1.0], fs=11.5)
    bul(s, [
        ("✓ 348,600フレーム全体で速度スパイク0 → 検出可能な3D ID入れ替わりなし。", 0, SUCCESS, True),
        ("✓ 残差はround-4より大幅改善（K_1_2：p90 3016→457、frac>5000 6.7%→0%）。", 0, SUCCESS, False),
        ("✓ 95–96%のフレームが安定した上面2D IDに紐付け。", 0, SUCCESS, False),
        ("カバレッジ85–87%（補間前）。欠けは遮蔽・未検出であり幾何の不良ではない。", 0, DARK, False),
        ("注意：滑らかな入れ替わりは指標に出ない → 交差の目視確認が受入基準。", 0, WARN, False),
    ], 0.5, 3.5, 7.2, 3.3, size=13.5)
    vid(s, VIDEOS / "3d_multiview_composite_clip.mp4", POSTERS / "3d_multiview_composite_clip.png",
        8.0, 1.45, 5.0, 4.2, caption="▶ 3D+マルチビュー合成（埋込）。Kグループ全montageはvideos/に。")

    # ---- 7. Segmentation
    s = blank(prs); title_bar(s, prs, "セグメンテーション — Detectron2 インスタンスマスク（幾何に依存しない識別手がかり）")
    bul(s, [
        ("動機：3D幾何は交差でIDが滑らかに入れ替わる。位置に由来しない識別手がかりが欲しかった。", 0, DARK, True),
        "Mask R-CNN（R50-FPN）を実験ごとにファインチューニング：上面1,001枚/2,551、側面870/2,454。",
        "複数ウィンドウで密推論 → 33,179クロップ（うちクリーン31,535）、171ビュー・トラックレット。",
        ("マスク→個体の対応付け（点がマスク内か）：", 0, ACCENT, True),
        ("トラックレット点がちょうど1つ内側 → クリーンな単個体クロップ", 1, DARK, False),
        ("0点 → 反射/誤検出（除外）。≥2点 → マスク融合（除外）。+ 面積スパイクQC", 1, DARK, False),
        "クロップを自己教師あり外見IDモデルへ（次スライド）。",
    ], 0.5, 1.35, 6.5, 5.5, size=14.5)
    img(s, PLOTS / "detectron2_vs_sleap_top.png", 7.2, 1.5, 5.8, 2.5, caption="上面 — Detectron2マスク（塗り）がSLEAP姿勢（黄）と一致。")
    img(s, PLOTS / "detectron2_vs_sleap_side1.png", 7.2, 4.4, 5.8, 2.4, caption="側面 — 分離時はクリーンな単個体マスク。")

    # ---- 8. Appearance ID
    s = blank(prs); title_bar(s, prs, "外見ID — 自己教師あり、人手ラベルなし")
    bul(s, [
        ("既存パイプラインから得られる2つの無償制約：", 0, DARK, True),
        ("must-link — 同一concreteトラックレット内のクロップは同一個体", 1, DARK, False),
        ("cannot-link — 同一フレーム内の2インスタンスは別個体", 1, DARK, False),
        ("自己教師の内部整合：cannot_link_satisfied = 0.99（IDラベル0）。", 0, ACCENT, True),
        ("主要な知見（PCA、全3方向）：", 0, DARK, True),
        ("✓ 1個体（妊娠したメス）が独自の弧として分離。", 1, SUCCESS, True),
        ("✗ 2オスは重なり合う（silhouette 0.31–0.43 ≪ フレーム毎ID識別に必要な0.91）。", 1, WARN, True),
        ("これは「メス1・類似オス2」という生物学とちょうど一致。", 0, GREY, False),
    ], 0.5, 1.35, 6.2, 5.6, size=14.5)
    img(s, PLOTS / "segid_per_view_pca.png", 6.7, 1.5, 6.4, 2.6, caption="方向別の埋め込み（PCA）：メス（橙）が分離、2オス（青/緑）は重なる。")
    img(s, PLOTS / "segid_fish_crops_top.png", 6.7, 4.55, 6.4, 2.0, caption="モデルへ入力する上面の魚体クロップ。")

    # ---- 9. Female anchor
    s = blank(prs); title_bar(s, prs, "メスアンカー — 達成、パイプラインに組込み済み", color=RGBColor(0x14, 0x6C, 0x2E))
    rows = [
        ["個体（幾何スロット）", "平均 P(メス)", "判定"],
        ["ID_2（slot 1）", "0.87", "メス"],
        ["ID_1（slot 0）", "0.08", "オス"],
        ["ID_3（slot 2）", "0.02", "オス"],
    ]
    tbl(s, rows, 0.5, 1.5, 5.4, col_w=[2.8, 1.5, 1.1], fs=13)
    bul(s, [
        ("✓ 個体単位の判定は決定的：信頼マージン 0.79（曖昧でない）。", 0, SUCCESS, True),
        ("✓ トラックレット単位で 91%メス / 97%オス を正しく側別（明瞭な二峰性）。", 0, SUCCESS, False),
        ("クロップ単位は0.72と中程度（妊娠状態のドリフトのため）だが、", 0, DARK, False),
        ("1個体あたり数百クロップを集約すると個体単位の性別判定は明確。", 1, DARK, False),
        ("実運用フィルタとして組込み → <tag>_3d_identified_sexed.csv（sex, female_prob）。", 0, ACCENT, True),
        ("フルレングスのオーケストレータ（script 40）を構築。K_1_1/K_1_3 の性別判定を2026-06-22に開始。", 0, GREY, False),
        ("メスは配偶者選択の「選ぶ側」 → 下流で直接取り出せるようになった。", 0, DARK, True),
    ], 0.5, 3.6, 6.4, 3.4, size=13.5)
    img(s, PLOTS / "female_prob_by_tracklet.png", 6.5, 1.5, 6.5, 4.0, caption="トラックレット毎のP(メス)：メス（赤）が右側に明瞭に分離。")

    # ---- 10. Males not separable
    s = blank(prs); title_bar(s, prs, "2オスは外見では区別できない（検証済み）", color=RGBColor(0x8A, 0x55, 0x00))
    bul(s, [
        ("これは二重の罠 — 両方の誤読を回避：", 0, DARK, True),
        ("罠1 — プールしたオス対オスはほぼ偶然（balacc 0.51–0.59、ARI≈0）。", 1, DARK, False),
        ("罠2 — ウィンドウ内のオス対オスは高く見える（最大1.0）— だがこれは位置の交絡。", 1, DARK, False),
        ("トラックレットのゲートが近接フレームを除外 → オスは常に空間的に離れる（<100 px は0%）、", 0, WARN, False),
        ("よってウィンドウ内モデルは外見でなく位置でカンニングできる。", 1, WARN, False),
        ("交絡除去テスト（背景マスク + ウィンドウホールドアウト + 位置ベースライン）：", 0, ACCENT, True),
        ("✗ CNN 0.235 vs 位置ベースライン 0.303 — どちらも偶然以下。オスの信号は消失。", 1, RGBColor(0xB0, 0x30, 0x30), True),
        ("結論：2オスは外見でなく 3D幾何 + 運動リゾルバ で解く。", 0, DARK, True),
        ("（オス外見の再挑戦は、ゲートしていない交差クロップ + 少量の人手ラベルがある場合のみ。）", 0, GREY, False),
    ], 0.6, 1.4, 12.1, 5.5, size=15)

    # ---- 11. Tracklet bind + resolver
    s = blank(prs); title_bar(s, prs, "交差を越えてIDを保持 — トラックレット↔3D紐付け + リゾルバ")
    bul(s, [
        ("保守的（stIFDゲート）トラックレットは固有の3Dスロットに紐付く：残差約6 px、マージン約209 px、全単射99.9%。", 0, DARK, True),
        "3D紐付けで単一視点曖昧フレームの約89–96%でIDを維持（Zは側面カメラから）。",
        ("924件のハードコアイベントが残存（3Dでペア<12 mm、全フレームの8.2%）：", 0, ACCENT, True),
        ("短時間857件（<0.5秒）・67件≥0.5秒・20件≥1秒が大半を占める", 1, DARK, False),
        ("運動連続性リゾルバ：513件解決（55.5%）、411件フラグ —", 0, DARK, False),
        ("うち18件のsuspect-swap=速度がリンカと矛盾=実IDエラーの可能性大（要優先レビュー）。", 1, WARN, True),
        ("次の一手：交差ウィンドウ内でのみメスアンカー+3D幾何を融合（resolve-at-crossing）。", 0, SUCCESS, True),
    ], 0.5, 1.35, 7.1, 5.6, size=14)
    img(s, PLOTS / "tracklet_coverage_stifd.png", 7.8, 1.5, 5.2, 2.4, caption="適応stIFDゲートでクリーンなカバレッジが約30%→約92%。")
    img(s, POSTERS / "tracklet_suspect_swaps18.png", 7.8, 4.35, 5.2, 2.3, caption="▶ videos/tracklet_suspect_swaps18.mp4 — 18件の要確認。")

    # ---- 12. Behaviour validation
    s = blank(prs); title_bar(s, prs, "トラッキングから生物学へ — 3D軌跡 vs 手動行動コーディング（BORIS）", color=RGBColor(0x14, 0x6C, 0x2E))
    bul(s, [
        ("K_1_2：手動コーディング89イベント（求愛54、競争26、交配4）。BORISのフレーム番号は", 0, DARK, True),
        ("我々の3Dフレーム（上面、166 fps）と1:1で一致 → 位置合わせ不要で直接重ね合わせ。", 1, DARK, False),
        ("メス=外見アンカー（ID_2）。2オスはダイアド（対）として検証（まだ区別不可）。", 0, GREY, False),
        ("✓ 幾何がコーディングを裏付け — メス–最近接オス距離が段階を反映：", 0, SUCCESS, True),
        ("ベースライン40 mm → 求愛28 mm → タッチ17 mm → ラッピング8 mm（接触）", 1, SUCCESS, True),
        ("✓ ラッピングの受容：受容28.0秒 vs 拒否3.6秒 — 研究室の独立した", 0, SUCCESS, False),
        ("~27秒 vs ~3秒の知見（results_EN_260616）と一致。イベント長のみから導出。", 1, SUCCESS, False),
        ("ループが閉じた：自動3D幾何が手動エソグラムを再現。", 0, ACCENT, True),
    ], 0.5, 1.4, 6.6, 5.5, size=14)
    img(s, BC / "behavior_distance_boxplot.png", 7.2, 1.55, 5.9, 5.1, caption="各コード行動中のメス–最近接オス距離。")

    # ---- 13. Spawn close-up
    s = blank(prs); title_bar(s, prs, "産卵イベント — 軌跡が交配を特定", color=RGBColor(0x14, 0x6C, 0x2E))
    img(s, BC / "spawn_closeup.png", 0.4, 1.3, 8.4, 5.7, caption="産卵 818.6秒：ラッピングを通じてメス–オス距離が接触まで低下し、直後に再開。")
    bul(s, [
        ("産卵時（t=818.6秒、フレーム135,883）：", 0, DARK, True),
        ("メスは約7–10 mmの持続接触", 1, DARK, False),
        ("オススロット ID_3（7.1 mm）vs ID_1（13.1 mm）と", 1, DARK, False),
        ("緑のラッピング区間を通じて。", 1, DARK, False),
        ("距離はラッピング中に接触まで低下し、", 0, DARK, False),
        ("産卵直後に再び開く。", 1, DARK, False),
        ("→ 軌跡はどのスロットが交配したかを示す。", 0, WARN, True),
        ("スロット→m1/m2（=父性）の対応付けには", 1, WARN, False),
        ("未解決のオスID問題が必要。", 1, WARN, False),
    ], 8.9, 1.5, 4.2, 5.3, size=13.5)

    # ---- 14. What's next
    s = blank(prs); title_bar(s, prs, "3D軌跡が次に解き明かすもの（results_EN_260616）")
    img(s, BC / "ethogram_distance_timeline.png", 0.4, 1.25, 8.7, 4.2,
        caption="全セッション：BORISエソグラム（上）と個体間距離（中）・メス速度（下）。")
    bul(s, [
        ("研究室の候補パラメータ — 我々の3D軌跡から算出可能：", 0, NAVY, True),
        ("✓ オス–メス距離の平均/分散（追従の安定性）— デモ済", 1, SUCCESS, False),
        ("✓ 受容/拒否前後のメス速度変調 — デモ済", 1, SUCCESS, False),
        ("→ 求愛中のクイックサークル半径・最大速度", 1, ACCENT, False),
        ("→ オス–メスの頭部方向アライメント（同期）指標", 1, ACCENT, False),
        ("→ 各求愛後のメス軌跡 = 受容 vs 拒否の信号", 1, ACCENT, False),
    ], 0.5, 5.65, 12.4, 1.7, size=13)

    # ---- 15. Summary
    s = blank(prs); title_bar(s, prs, "まとめと次のステップ")
    bul(s, [
        ("パイプラインの現状：", 0, NAVY, True),
        ("✓ キャリブレーション — 実運用モデル検証済（0.90 mm）。剛体リファインメント経路も準備済。", 0, SUCCESS, False),
        ("✓ SLEAP round-5 を実運用 — 348kフレームでクリーンな3D、速度スパイク0。", 0, SUCCESS, False),
        ("✓ 3次元再構成 — 幾何は堅牢。安定した上面アンカーでIDを95–96%紐付け。", 0, SUCCESS, False),
        ("✓ メスアンカー — メスを確実にF/M判定し、性別付きCSV出力に組込み。", 0, SUCCESS, False),
        ("✓ 行動検証 — 3D軌跡が手動BORISエソグラムを再現（距離段階 + ラッピング受容/拒否）。", 0, SUCCESS, False),
        ("✗ 外見による2オス区別 — 区別不可と確認。幾何・連続性の問題。", 0, WARN, False),
        ("次：", 0, NAVY, True),
        ("→ resolve-at-crossing：交差ウィンドウ内でメスアンカー+3Dを融合。18件のsuspect-swapを目視確認。", 1, ACCENT, False),
        ("→ round-6 SLEAPを近接ラベルで学習（ウォームスタート）し、根本から共有尾の入れ替わりを低減。", 1, ACCENT, False),
        ("→ 研究室の配偶者選択指標を3Dから算出（追従の安定性、頭部同期、サークル半径）。K_1_1/K_1_3へ展開。", 1, ACCENT, False),
    ], 0.6, 1.45, 12.1, 5.6, size=14.5)

    apply_jp_font(prs, JP_FONT)
    out = HERE / "medaka_progress_report_20260622_JA.pptx"
    prs.save(str(out))
    return out


if __name__ == "__main__":
    out = build()
    print(f"Japanese deck written: {out}  ({out.stat().st_size/1e6:.1f} MB)")
