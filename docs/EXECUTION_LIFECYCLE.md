# Recommendation and execution lifecycle

An analysis run proposes tenant-scoped recommendation records. Each starts as `pending_approval`. The UI exposes action rationale, eligibility, MCIA components and review controls. Approval/rejection requires Owner, Admin, or Approver authorization and records reviewer, decision, note, and audit entry.

The approval endpoint resumes the LangGraph checkpoint for that run. The Execute agent checks recommendation tenant, customer tenant, current reviewer membership and role, approval record, recommendation state, and existing execution. A unique database constraint on recommendation execution prevents repeats. An approved action creates a clearly labelled simulation record; no CRM, messaging, billing, or support service is contacted. Rejected recommendations cannot execute.

An execution is not a customer outcome. Authorized analysts can later record manually observed follow-up (`retained`, `churned`, `unknown`, or `no_response`); these reports do not estimate intervention effect or ROI.
