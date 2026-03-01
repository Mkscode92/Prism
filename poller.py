from __future__ import annotations

import asyncio
import functools
import logging
from datetime import datetime, timedelta, timezone

from google_play_scraper import Sort, reviews

logger = logging.getLogger("prism.poller")

# Tracks review IDs already seen (in-memory, resets on restart)
_seen_ids: set[str] = set()

# How far back to look for reviews — slightly wider than poll interval to avoid gaps
_WINDOW_MINUTES = 15


def _fetch_reviews(package_name: str) -> list[dict]:
    """Synchronous fetch — run in executor so it doesn't block the event loop."""
    result, _ = reviews(
        package_name,
        lang="en",
        country="us",
        sort=Sort.NEWEST,
        count=30,
    )
    return result


def _is_within_window(r: dict) -> bool:
    """Return True if the review was posted within _WINDOW_MINUTES."""
    at = r.get("at")
    if not isinstance(at, datetime):
        return False
    if at.tzinfo is None:
        at = at.replace(tzinfo=timezone.utc)
    return at >= datetime.now(timezone.utc) - timedelta(minutes=_WINDOW_MINUTES)


async def poll_loop(
    app_registry: dict[str, str],
    on_new_review,       # async callback(package_name, repo_url, raw_review_dict)
    interval: int = 300, # seconds between polls (default 10 min)
) -> None:
    """
    Background task: every `interval` seconds, fetch the 5 worst recent reviews
    (score < 3, posted within the last 10 minutes) for every registered app and
    pass each new one to on_new_review. Uses _seen_ids to avoid duplicates.
    """
    logger.info(f"Poller started — checking every {interval // 10} minutes for score < 3 reviews", interval)
    # Short delay so the server is fully ready before the first poll
    await asyncio.sleep(10)

    while True:
        loop = asyncio.get_running_loop()
        for package_name, repo_url in list(app_registry.items()):
            try:
                result = await loop.run_in_executor(
                    None, functools.partial(_fetch_reviews, package_name)
                )

                # Keep only low-rated reviews posted within the window
                candidates = [
                    r for r in result
                    if r.get("score", 5) < 3 and _is_within_window(r)
                ]

                # Sort worst-first, take the 5 most actionable
                candidates.sort(key=lambda r: r.get("score", 5))
                candidates = candidates[:5]

                added = 0
                for r in candidates:
                    rid = r["reviewId"]
                    if rid in _seen_ids:
                        continue
                    _seen_ids.add(rid)
                    added += 1
                    await on_new_review(package_name, repo_url, r)

                if added:
                    logger.info(
                        "Poller: %d new low-rated review(s) queued for %s",
                        added, package_name,
                    )
            except Exception:
                logger.exception("Poller: failed to fetch reviews for %s", package_name)

        await asyncio.sleep(interval)
