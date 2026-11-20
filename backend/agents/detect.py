from datetime import datetime, timezone
from backend.agents.schemas import DetectResult, RiskSignal


def run_detect(customer_id: str, features: dict, events: list[dict]) -> DetectResult:
    rules = [
        ("payment_failures_90d", lambda v: v > 0, "Payment failures are present in supplied aggregate data."),
        ("support_tickets_30d", lambda v: v >= 3, "Support contact volume is elevated in supplied aggregate data."),
        ("unresolved_tickets", lambda v: v > 0, "Unresolved support tickets are present in supplied aggregate data."),
        ("last_activity_days", lambda v: v >= 14, "Recent activity gap is long relative to the 30-day window."),
        ("login_frequency_30d", lambda v: v <= 5, "Login frequency is low in the 30-day aggregate."),
        ("service_downtime_hours", lambda v: v >= 5, "Service downtime is elevated in the supplied aggregate."),
        ("satisfaction_score", lambda v: v <= 2.5, "Satisfaction score is low in the supplied aggregate."),
    ]
    signals = []
    for field, predicate, description in rules:
        value = features.get(field)
        if value is not None and predicate(float(value)):
            signals.append(RiskSignal(signal=description, severity="high" if field in ("payment_failures_90d", "unresolved_tickets") else "medium",
                supporting_feature=field, observed_value=str(value), evidence_strength="aggregate feature observed",
                detected_at=datetime.now(timezone.utc).isoformat()))
    for event in events:
        if event.get("severity") == "high":
            signals.append(RiskSignal(signal=f"Simulated {event['event_type']} event", severity="medium",
                supporting_feature=event.get("indicator", "event"), observed_value=event["timestamp"],
                evidence_strength="simulated event derived from aggregate feature", detected_at=datetime.now(timezone.utc).isoformat()))
    severe = sum(item.severity == "high" for item in signals)
    overall = "critical" if severe >= 2 else "high" if severe == 1 or len(signals) >= 4 else "medium" if signals else "low"
    return DetectResult(customer_id=customer_id, signals=signals, overall_severity=overall,
                        summary=f"Detected {len(signals)} evidence-backed signal(s); simulated events are identified separately.")
