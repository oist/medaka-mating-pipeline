import pytest

from scripts.common.experiment_sets import resolve_experiments

FOLDERS = [
    "20260112_P_1_1_FCMC",
    "20260113_P_1_2_FCMC",  # same set, different date
    "20260112_P_1_3_FCMC",
    "20260118_P_2_1_FCMC",
    "20260118_P_3_1_FCMC",
    "20260113_M_1_2_FCMC_P2",
    "20260113_M_1_2_FCMC",
    "cali_20251217",
]


@pytest.fixture
def data_dir(tmp_path):
    for name in FOLDERS:
        (tmp_path / name).mkdir()
    return tmp_path


def test_set_token_matches_group_and_trials_across_dates(data_dir):
    assert resolve_experiments(["P12"], data_dir) == [
        "20260112_P_1_1_FCMC",
        "20260112_P_1_3_FCMC",
        "20260113_P_1_2_FCMC",
        "20260118_P_2_1_FCMC",
    ]


def test_part2_folders_excluded_unless_requested(data_dir):
    assert resolve_experiments(["M1"], data_dir) == ["20260113_M_1_2_FCMC"]
    assert "20260113_M_1_2_FCMC_P2" in resolve_experiments(["M1"], data_dir, include_part2=True)


def test_all_and_dedup(data_dir):
    names = resolve_experiments(["20260118_P_3_1_FCMC", "ALL"], data_dir)
    assert names[0] == "20260118_P_3_1_FCMC"
    assert len(names) == len(set(names)) == 6
    assert "cali_20251217" not in names


def test_unknown_tokens_raise(data_dir):
    with pytest.raises(FileNotFoundError):
        resolve_experiments(["Q12"], data_dir)
    with pytest.raises(FileNotFoundError):
        resolve_experiments(["20990101_P_1_1_FCMC"], data_dir)
    with pytest.raises(ValueError):
        resolve_experiments(["not-a-token"], data_dir)
