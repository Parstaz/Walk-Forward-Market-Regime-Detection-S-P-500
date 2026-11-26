import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src  # noqa: E402,F401  (pins BLAS threads before numpy loads)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pandas_market_calendars as mcal  # noqa: E402
import pytest  # noqa: E402

from src.pipeline import Paths, build_history  # noqa: E402

# Small settings so a full build takes seconds instead of minutes.
BURN_IN = 250
REFIT_EVERY = 10
N_SESSIONS = 600
BUILD_AT = 560  # history is first built through session index BUILD_AT - 1


def make_prices(n=N_SESSIONS, seed=0) -> pd.DataFrame:
    """Synthetic adjusted closes on real NYSE sessions with calm and stressed spells."""
    sessions = mcal.get_calendar("NYSE").valid_days("2019-01-01", "2023-12-31").tz_localize(None)[:n]
    rng = np.random.default_rng(seed)
    vol = np.where((np.arange(n) // 60) % 3 == 2, 0.025, 0.008)
    r = rng.normal(0.0003, vol)
    close = 100 * np.exp(np.cumsum(r))
    return pd.DataFrame({"Close": close}, index=pd.DatetimeIndex(sessions, name="Date"))


class FakeFetch:
    """Stands in for fetch_prices: serves the truth series up to `cutoff`.

    `scale` multiplies every served price, like yfinance re-adjusting the
    whole history after a dividend. `edit` can alter the served frame.
    """

    def __init__(self, truth, cutoff, scale=1.0, edit=None):
        self.truth, self.cutoff, self.scale, self.edit = truth, cutoff, scale, edit
        self.calls = []

    def __call__(self, start=None, **kwargs):
        self.calls.append(start)
        df = self.truth.loc[pd.Timestamp(start or "1900-01-01"): self.cutoff] * self.scale
        return self.edit(df.copy()) if self.edit else df

    @property
    def now(self):
        return pd.Timestamp(self.cutoff).tz_localize("America/New_York") + pd.Timedelta(hours=18)


@pytest.fixture(scope="session")
def truth():
    return make_prices()


@pytest.fixture(scope="session")
def built_template(tmp_path_factory, truth):
    """History built through session BUILD_AT - 1, shared read-only by tests."""
    root = tmp_path_factory.mktemp("built")
    paths = Paths(root / "data", root / "models")
    paths.data_dir.mkdir()
    shutil.copy(ROOT / "data" / "nber_recessions.csv", paths.nber)
    fetch = FakeFetch(truth, truth.index[BUILD_AT - 1])
    build_history(paths, fetch=fetch, now=fetch.now, burn_in=BURN_IN, refit_every=REFIT_EVERY)
    return root


@pytest.fixture
def built(tmp_path, built_template):
    """A private copy of the built history."""
    shutil.copytree(built_template, tmp_path / "h")
    return Paths(tmp_path / "h" / "data", tmp_path / "h" / "models")


def snapshot(paths: Paths) -> dict:
    """Bytes of every file under data/ and models/, keyed by relative path."""
    files = {}
    for d in (paths.data_dir, paths.models_dir):
        for f in sorted(d.rglob("*")):
            if f.is_file():
                files[str(f.relative_to(d.parent))] = f.read_bytes()
    return files
