"""Cached access to artifacts and cached scenario calculations.

Heavy work happens once in scripts/build_artifacts.py. Here everything is either read from
disk once per server (cache_resource / cache_data) or recomputed only when slider inputs change.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, TypeVar

import pandas as pd
import streamlit as st

from clis import artifacts as art
from clis.config import ARTIFACT_DIR, SETTINGS, Economics, Settings
from clis.errors import ArtifactError, CLIError, StrategyError
from clis.model.scoring import ScoringModel

T = TypeVar("T")


def guard(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run fn; on a known project error show a clear message and stop the page."""
    try:
        return fn(*args, **kwargs)
    except ArtifactError as exc:
        st.error(f"**Model files unavailable.** {exc}")
        st.stop()
    except CLIError as exc:
        st.error(f"**Could not compute this view.** {exc}")
        st.stop()


@st.cache_resource(show_spinner=False)
def model() -> ScoringModel:
    return ScoringModel.load(ARTIFACT_DIR)


@st.cache_data(show_spinner=False)
def table(name: str) -> pd.DataFrame:
    return art.load_table(name)


@st.cache_data(show_spinner=False)
def summary() -> dict[str, Any]:
    return art.load_json("summary")


@st.cache_data(show_spinner=False)
def split(part: str) -> pd.DataFrame:
    acct = table("accounts")
    return acct[acct["split"] == part].reset_index(drop=True)


def settings_with(econ: Economics) -> Settings:
    return replace(SETTINGS, economics=econ)


@st.cache_data(show_spinner=False)
def pli_scenario(apr: float, lgd: float, ch_util: float, cp_util: float,
                 cutoff: float | None) -> dict[str, Any]:
    """Re-tune (on validation) and re-evaluate (on holdout) for the chosen assumptions.
    cutoff=None means 'use the profit-maximizing cut-off'."""
    from clis.strategy import pli
    cfg = settings_with(replace(SETTINGS.economics, apr=apr, lgd=lgd))
    valid, holdout = split("valid"), split("holdout")
    try:
        choice = pli.choose_cutoff(valid, cfg, ch_util, cp_util)
        best, reason = choice.cutoff, choice.reason
        sweep = choice.sweep
    except StrategyError as exc:
        best, reason, sweep = None, str(exc), pli.sweep_cutoffs(valid, cfg, ch_util)
    used = cutoff if cutoff is not None else best
    if used is None:
        raise StrategyError(reason)
    champ_m, champ = pli.champion_result(holdout, cfg, cp_util)
    chall_m, chall = pli.challenger_result(holdout, cfg, used, ch_util)
    return {"best": best, "used": used, "reason": reason, "sweep": sweep,
            "holdout_sweep": pli.sweep_cutoffs(holdout, cfg, ch_util),
            "champion": champ, "challenger": chall,
            "swap": pli.swap_sets(holdout, champ_m, chall_m)}
