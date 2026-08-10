"""Time-varying universe reconstitution with hysteresis.

Smoothly transitions the universe over time by applying a buffer zone
around the selection boundary.  Stocks inside the buffer are not flipped
in or out unless their score moves decisively, which avoids excessive
turnover from borderline selections.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .._preconditions import require

__all__ = [
    "UniverseReconstitution",
]


@dataclass
class UniverseReconstitution:
    """Manage monthly / quarterly universe changes with hysteresis.

    Parameters
    ----------
    buffer_pct : float
        Fraction of ``n_stocks`` used as the hysteresis band.  For
        example, with ``n_stocks=50`` and ``buffer_pct=0.10``, stocks
        ranked 46–55 are in the buffer zone: incumbents stay, outsiders
        don't enter.
    max_turnover_pct : float
        Maximum fraction of the universe that can change in a single
        reconstitution (0.0–1.0).
    """

    buffer_pct: float = 0.10
    max_turnover_pct: float = 0.30

    def __post_init__(self) -> None:
        require(0.0 <= self.buffer_pct < 0.5, "buffer_pct must be in [0, 0.5)")
        require(
            0.0 < self.max_turnover_pct <= 1.0, "max_turnover_pct must be in (0, 1]"
        )

    def reconstitute(
        self,
        current_universe: list[str],
        scores: pd.Series,
        n_stocks: int,
    ) -> list[str]:
        """Update the universe with hysteresis to limit turnover.

        Parameters
        ----------
        current_universe : list of str
            Tickers currently in the universe.
        scores : Series
            SPT composite score for every candidate (higher = better).
            Must include all current universe members that are still
            scorable.
        n_stocks : int
            Target universe size.

        Returns
        -------
        list of str
            Updated universe.
        """
        scores = scores.dropna().sort_values(ascending=False)
        all_ranked = list(scores.index)

        if not current_universe:
            return all_ranked[:n_stocks]

        buffer = max(1, int(n_stocks * self.buffer_pct))
        upper = n_stocks + buffer
        max_changes = max(1, int(n_stocks * self.max_turnover_pct))

        # Stocks ranked in the top ``n_stocks`` are definite keeps / adds
        definite_in = set(all_ranked[:n_stocks])
        # Stocks ranked in the buffer zone keep their status quo
        buffer_zone = (
            set(all_ranked[n_stocks:upper]) if upper <= len(all_ranked) else set()
        )

        current_set = set(current_universe) & set(all_ranked)

        # Start with definite-in stocks
        new_universe = set(definite_in)

        # Incumbent stocks that landed in the buffer zone stay —
        # they displace the worst-ranked definite-in non-incumbents.
        buffer_incumbents = current_set & buffer_zone
        for t in buffer_incumbents:
            new_universe.add(t)

        # Trim to target size, but prefer keeping incumbents over newcomers.
        if len(new_universe) > n_stocks:
            ranked_new = [t for t in all_ranked if t in new_universe]
            keep: set[str] = set()
            for t in ranked_new:
                if len(keep) >= n_stocks:
                    break
                keep.add(t)
                # If a buffer incumbent hasn't been added yet, force it in
                # by not counting a later non-incumbent.
            # Ensure all buffer incumbents are kept
            for t in buffer_incumbents:
                keep.add(t)
            if len(keep) > n_stocks:
                # Drop lowest-ranked non-incumbent newcomers
                newcomers = [
                    t
                    for t in reversed(ranked_new)
                    if t in keep and t not in current_set
                ]
                for nc in newcomers:
                    if len(keep) <= n_stocks:
                        break
                    keep.discard(nc)
            new_universe = keep

        # Enforce max turnover: limit number of changes vs current set
        additions = new_universe - current_set
        removals = current_set - new_universe

        if len(additions) + len(removals) > 2 * max_changes:
            # Keep as many incumbents as possible, only swap the clearest upgrades
            additions_ranked = [t for t in all_ranked if t in additions]
            removals_ranked = [t for t in reversed(all_ranked) if t in removals]

            n_swap = min(max_changes, len(additions_ranked), len(removals_ranked))
            kept_additions = set(additions_ranked[:n_swap])
            forced_removals = set(removals_ranked[:n_swap])

            new_universe = (current_set - forced_removals) | kept_additions

        # Final safety: ensure we have at most n_stocks, sorted by score
        ordered = [t for t in all_ranked if t in new_universe][:n_stocks]
        return ordered

    def reconstitute_timeseries(
        self,
        scores_by_date: dict[pd.Timestamp, pd.Series],
        n_stocks: int,
        initial_universe: list[str] | None = None,
    ) -> dict[pd.Timestamp, list[str]]:
        """Apply reconstitution across a chronological series of scores.

        Parameters
        ----------
        scores_by_date : dict
            Mapping from date → full score series.
        n_stocks : int
            Target universe size.
        initial_universe : list of str, optional
            Starting universe.  If ``None``, the first date uses the top
            ``n_stocks`` directly.

        Returns
        -------
        dict mapping date → list of selected tickers.
        """
        result: dict[pd.Timestamp, list[str]] = {}
        current = initial_universe or []

        for dt in sorted(scores_by_date):
            current = self.reconstitute(current, scores_by_date[dt], n_stocks)
            result[dt] = current

        return result
