"""Transparent M/T/R/F ranking; score is utility, not retention probability."""
from copy import deepcopy

DEFAULT_WEIGHTS = {"money": 0.25, "time": 0.25, "relation": 0.25, "feasibility": 0.25}
ACTION_LIBRARY = [
    {"id":"billing_assistance","name":"Billing assistance","money":2,"time":2,"relation":4,"feasibility":5,"approval_required":True,"description":"Review billing friction and payment options."},
    {"id":"priority_support","name":"Priority technical support","money":2,"time":2,"relation":5,"feasibility":4,"approval_required":True,"description":"Escalate unresolved technical issues."},
    {"id":"onboarding","name":"Product onboarding","money":2,"time":3,"relation":4,"feasibility":4,"approval_required":True,"description":"Offer a guided product-use session."},
    {"id":"success_followup","name":"Customer-success follow-up","money":1,"time":1,"relation":4,"feasibility":5,"approval_required":True,"description":"Arrange a customer-success check-in."},
    {"id":"retention_discount","name":"Retention discount","money":5,"time":1,"relation":3,"feasibility":3,"approval_required":True,"description":"Offer a reviewed discount where policy permits."},
    {"id":"contract_review","name":"Contract review","money":1,"time":3,"relation":3,"feasibility":4,"approval_required":True,"description":"Review renewal timing and contract fit."},
]


def rank_actions(features: dict, weights: dict | None = None) -> list[dict]:
    configured = deepcopy(weights or DEFAULT_WEIGHTS)
    if set(configured) != set(DEFAULT_WEIGHTS) or any(float(v) < 0 for v in configured.values()):
        raise ValueError("Weights must be non-negative M/T/R/F values")
    total = sum(map(float, configured.values()))
    if total <= 0: raise ValueError("At least one MCIA weight must be positive")
    configured = {key: float(value)/total for key, value in configured.items()}
    candidates = []
    for action in ACTION_LIBRARY:
        # Ratings use 1–5. High expense/time is undesirable; relation/feasibility are positive.
        m, t, r, f = (action[key] for key in ("money", "time", "relation", "feasibility"))
        score = configured["relation"]*r/5 + configured["feasibility"]*f/5 - configured["money"]*m/5 - configured["time"]*t/5
        rationale = "Selected from the configured M/T/R/F utility ratings; review customer evidence before approval."
        candidates.append({**action, "mcia_score": round(score, 4), "weights": configured, "rationale": rationale,
                           "eligibility": "eligible", "score_interpretation": "relative action utility; not retention probability"})
    return sorted(candidates, key=lambda item: item["mcia_score"], reverse=True)
