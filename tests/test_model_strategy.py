from dataclasses import replace

import numpy as np
import pytest

from clis.config import SETTINGS
from clis.decision.rli import decide_portfolio
from clis.errors import ArtifactError
from clis.features.builder import build_features
from clis.model.scoring import ScoringModel
from clis.model.train import split_accounts, train_and_select
from clis.monitoring.stability import psi
from clis.strategy.economics import tiered_increase
from clis.strategy.pli import choose_cutoff


@pytest.fixture
def trained(raw_accounts):
    acct = build_features(raw_accounts)
    split = split_accounts(acct)
    model, _ = train_and_select(split)
    for df in (split.train, split.valid, split.holdout):
        df["pd"] = model.predict_pd(df)
    return model, split


def test_model_roundtrip(trained, tmp_path):
    model, split = trained
    model.save(tmp_path)
    loaded = ScoringModel.load(tmp_path)
    np.testing.assert_allclose(model.predict_pd(split.holdout), loaded.predict_pd(split.holdout), rtol=1e-6)


def test_missing_artifact_raises(tmp_path):
    with pytest.raises(ArtifactError):
        ScoringModel.load(tmp_path)


def test_split_is_disjoint(trained):
    _, s = trained
    ids = [set(df.account_id) for df in (s.train, s.valid, s.holdout)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])


def test_cutoff_tuned_without_holdout(trained):
    _, s = trained
    cfg = replace(SETTINGS, policy=replace(SETTINGS.policy, min_segment_size=5,
                                           champion_min_util=0.0, challenger_min_util=0.0))
    a = choose_cutoff(s.valid, cfg).cutoff
    s.holdout = s.holdout.assign(default_next_month=1 - s.holdout.default_next_month)
    assert choose_cutoff(s.valid, cfg).cutoff == a   # holdout labels cannot move the choice


def test_rli_rules_and_reasons(trained):
    model, s = trained
    t = s.holdout.copy()
    t.loc[t.index[0], "pay_status_1"] = 2
    d = decide_portfolio(t, model, SETTINGS, cutoff=0.99)
    assert d.iloc[0].rli_decision == "DECLINE" and "R01" in d.iloc[0].rli_codes
    declined_by_score = decide_portfolio(t, model, SETTINGS, cutoff=0.0)
    score_only = declined_by_score[declined_by_score.rli_score_decline]
    assert score_only.rli_codes.str.contains("R1").all()   # specific driver reasons, never blank


def test_tiered_increase_rewards_lower_risk():
    inc = tiered_increase(np.array([100_000, 100_000, 100_000]), np.array([0.02, 0.12, 0.5]), SETTINGS.sizing)
    assert inc[0] > inc[1] > inc[2]
    assert inc.max() <= SETTINGS.sizing.cap


def test_psi_behaviour():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 5000)
    assert psi(a, a) < 1e-6
    assert psi(a, a + 1.0) > 0.25
