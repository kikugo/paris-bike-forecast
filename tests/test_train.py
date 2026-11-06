from forecast.train import decide

CHAMP = {"version": "1", "mae": 20.0, "snapshot": "abc", "params": {"n_estimators": 600}}


def _no_rescore(params):
    raise AssertionError("should not re-score on the same snapshot")


def test_first_model_becomes_champion():
    assert decide(25.0, "abc", None, _no_rescore)["promote"]


def test_worse_or_equal_model_is_not_promoted():
    assert not decide(20.0, "abc", CHAMP, _no_rescore)["promote"]
    assert not decide(21.0, "abc", CHAMP, _no_rescore)["promote"]


def test_better_model_is_promoted():
    d = decide(19.5, "abc", CHAMP, _no_rescore)
    assert d["promote"] and d["champion_mae"] == 20.0 and not d["champion_rescored"]


def test_champion_from_another_snapshot_is_rescored_first():
    calls = []

    def rescore(params):
        calls.append(params)
        return 18.0

    d = decide(19.0, "new", CHAMP, rescore)
    assert calls == [CHAMP["params"]]
    assert d["champion_rescored"] and d["champion_mae"] == 18.0
    assert not d["promote"]
