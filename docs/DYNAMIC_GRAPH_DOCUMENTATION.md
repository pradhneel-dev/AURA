# Events and graph (current prototype)

The source CSVs contain aggregate features, not timestamped event sequences. `backend/services/event_simulator.py` creates seeded events for payment failures, support tickets, API errors and logins. Counts are bounded and derived from corresponding aggregate feature values; dates are simulated relative to a fixed UTC demo anchor. Every record carries synthetic provenance.

`backend/graph_engine/customer_graph.py` builds a NetworkX graph with customer, event and derived indicator nodes. The API exposes events and graph per customer. Graph links describe generated event associations only; they do not establish causal relationships. Additional event types in the project brief are not yet simulated.
