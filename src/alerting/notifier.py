"""Slack alerting for brand visibility drops."""
import json
import os
import urllib.request
from dataclasses import dataclass

SLACK_WEBHOOK_URL      = os.getenv("SLACK_WEBHOOK_URL", "")
VISIBILITY_THRESHOLD   = float(os.getenv("VISIBILITY_ALERT_THRESHOLD", "15.0"))
DROP_THRESHOLD         = float(os.getenv("VISIBILITY_DROP_ALERT", "5.0"))


@dataclass
class AlertResult:
    brand: str
    current: float
    previous: float | None
    triggered: bool
    reason: str


def _post_slack(text: str) -> None:
    if not SLACK_WEBHOOK_URL:
        print("[notifier] SLACK_WEBHOOK_URL not set — skipping notification.")
        return
    payload = json.dumps({"text": text}).encode()
    req = urllib.request.Request(
        SLACK_WEBHOOK_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    urllib.request.urlopen(req, timeout=10)


def _build_message(alerts: list[AlertResult], category: str) -> str:
    lines = [f":warning: *Brand Visibility Alert* — category: `{category}`\n"]
    for a in alerts:
        if "below threshold" in a.reason:
            lines.append(
                f"• *{a.brand}*: visibility dropped to *{a.current:.1f}%* "
                f"(threshold: {VISIBILITY_THRESHOLD:.0f}%)"
            )
        else:
            drop = a.previous - a.current
            lines.append(
                f"• *{a.brand}*: visibility fell *{drop:.1f}pp* "
                f"({a.previous:.1f}% → {a.current:.1f}%)"
            )
    return "\n".join(lines)


def check_and_alert(category: str, brands: list[str]) -> list[AlertResult]:
    """
    Query latest visibility for each brand in the category.
    Fires a Slack alert if any brand is below VISIBILITY_THRESHOLD
    or dropped more than DROP_THRESHOLD points vs the previous run.
    Returns list of AlertResult (all brands, not just triggered ones).
    """
    from src.metrics.calculator import visibility_summary_by_category, _tbl
    from src.storage.store import query_df

    df = visibility_summary_by_category(category, brands)
    if df.empty:
        return []

    # Previous run visibility — compare last two distinct run dates
    bm = _tbl("brand_mentions")
    pr = _tbl("prompts")
    prev_sql = f"""
    SELECT bm.brand, AVG(bm.position) AS avg_pos, COUNT(*) AS cnt
    FROM {bm} bm
    JOIN {pr} p ON bm.prompt_id = p.prompt_id
    WHERE p.category = '{category}'
      AND DATE(bm.created_at) = (
          SELECT DISTINCT DATE(created_at) FROM {bm}
          ORDER BY DATE(created_at) DESC
          LIMIT 1 OFFSET 1
      )
    GROUP BY bm.brand
    """
    try:
        prev_df = query_df(prev_sql)
        prev_map = dict(zip(prev_df["brand"], prev_df["cnt"])) if not prev_df.empty else {}
    except Exception:
        prev_map = {}

    results: list[AlertResult] = []
    triggered: list[AlertResult] = []

    for _, row in df.iterrows():
        brand = row["brand"]
        current = float(row["visibility_pct"])
        previous = None
        reason = ""
        fired = False

        if current < VISIBILITY_THRESHOLD:
            fired = True
            reason = "below threshold"

        if brand in prev_map:
            total_prev = sum(prev_map.values()) or 1
            prev_vis = prev_map[brand] / total_prev * 100
            previous = round(prev_vis, 1)
            if previous - current >= DROP_THRESHOLD:
                fired = True
                reason = reason or "significant drop"

        ar = AlertResult(brand=brand, current=current, previous=previous,
                         triggered=fired, reason=reason)
        results.append(ar)
        if fired:
            triggered.append(ar)

    if triggered:
        msg = _build_message(triggered, category)
        _post_slack(msg)
        print(f"[notifier] Alert sent for {len(triggered)} brand(s).")
    else:
        print("[notifier] All brands within normal range — no alert sent.")

    return results
