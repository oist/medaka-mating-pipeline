"""Unit tests for the female-anchor production filter (pure logic)."""

import pandas as pd

from sam2_id_experiment.src.id_model.female_anchor import (
    aggregate_female_prob, assign_single_female, sex_labels,
)


def test_assign_single_female_picks_argmax():
    call = assign_single_female({"ID_1": 0.1, "ID_2": 0.92, "ID_3": 0.08},
                                n_crops={"ID_1": 100, "ID_2": 120, "ID_3": 90})
    assert call.female_fish == "ID_2"
    assert not call.ambiguous
    assert call.confidence_margin > 0.7
    assert call.n_crops["ID_2"] == 120


def test_exactly_one_female():
    probs = {0: 0.7, 1: 0.6, 2: 0.05}
    call = assign_single_female(probs)
    labels = sex_labels(probs, call.female_fish)
    assert sum(v == "F" for v in labels.values()) == 1
    assert labels[0] == "F" and labels[1] == "M" and labels[2] == "M"


def test_ambiguous_flag_low_margin():
    # two fish nearly tied for female -> flagged for review
    call = assign_single_female({0: 0.55, 1: 0.52, 2: 0.04}, margin_threshold=0.15)
    assert call.ambiguous and call.confidence_margin < 0.15


def test_aggregate_female_prob():
    df = pd.DataFrame({
        "fish": ["A", "A", "A", "B", "B"],
        "female_prob": [0.9, 0.8, 1.0, 0.1, 0.2],
    })
    agg = aggregate_female_prob(df, "fish").set_index("fish")
    assert abs(agg.loc["A", "mean_female_prob"] - 0.9) < 1e-6
    assert agg.loc["A", "n_crops"] == 3
    assert abs(agg.loc["B", "mean_female_prob"] - 0.15) < 1e-6


def test_empty():
    call = assign_single_female({})
    assert call.female_fish is None and call.ambiguous
