"""Seeded, explicitly synthetic event generation guided by aggregate customer features."""
from datetime import datetime, timedelta, timezone
import random

ANCHOR = datetime(2026, 11, 5, 12, 0, tzinfo=timezone.utc)


def simulate_events(customer_id: str, features: dict, seed: int = 42) -> list[dict]:
    rng = random.Random(f"{seed}:{customer_id}")
    specs = [
        ("payment_failure", min(5, int(features.get("payment_failures_90d", 0))), "payment_failures_90d", "high"),
        ("support_ticket", min(8, int(features.get("support_tickets_30d", 0))), "support_tickets_30d", "medium"),
        ("api_error", min(6, int(float(features.get("api_error_rate", 0)) // 3)), "api_error_rate", "medium"),
        ("login", min(12, int(float(features.get("login_frequency_30d", 0)) // 5)), "login_frequency_30d", "low"),
    ]
    generated = []
    for event_type, count, indicator, severity in specs:
        for index in range(count):
            days_back = rng.randint(0, 29)
            timestamp = ANCHOR - timedelta(days=days_back, minutes=rng.randint(0, 1439))
            generated.append({"event_id": f"SIM-{customer_id}-{event_type}-{index+1}", "customer_id": customer_id,
                              "event_type": event_type, "timestamp": timestamp.isoformat(),
                              "payload": {"aggregate_feature": indicator, "event_index": index+1},
                              "indicator": indicator, "severity": severity, "source": "AURA event simulator",
                              "provenance": "simulated; not present in supplied CSV"})
    generated.sort(key=lambda event: event["timestamp"])
    return generated
