# MCIA action ranking (current prototype)

Candidate actions use 1–5 ratings for Money, Time, Relation and Feasibility. High Money and Time are costs; Relation and Feasibility are benefits. The implementation normalizes nonnegative configured weights to sum to one and calculates `R*wR + F*wF - M*wM - T*wT`, after dividing each rating by five. The score is relative utility, not retention probability or ROI.

The action library and configurable tenant weights are in `backend/mcia/optimizer.py` and tenant settings. Candidate eligibility and policy checks are applied before workflow proposals. Recommendations persist with rationale, eligibility, score components, and approval state. Owner/Admin users configure weights; Approver/Owner/Admin roles review proposals. Approved execution is simulation only. This framework does not optimize from manually entered outcomes and does not imply causal effect or ROI.
