"""
Compatibility wrapper for the Research Lab subsystem.

The implementation now lives in backend/research_lab/engine.py so the
feature has a dedicated package and clearer ownership boundaries.
"""

from research_lab.engine import (  # noqa: F401
    DEFAULT_FREQUENCY,
    DEFAULT_START_DATE,
    DEFAULT_SYMBOLS,
    REGIME_LABELS,
    ForecastResult,
    ResearchLab,
)


if __name__ == '__main__':
    import json

    lab = ResearchLab(dbClient=None)
    result = lab.run(persist=False, useSyntheticData=True, verbose=True)
    print(json.dumps(result, indent=2))
