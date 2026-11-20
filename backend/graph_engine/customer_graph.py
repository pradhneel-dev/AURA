"""Construct a descriptive customer-event-indicator graph from synthetic events."""
import networkx as nx


def build_customer_graph(customer_id: str, events: list[dict]) -> dict:
    graph = nx.Graph()
    graph.add_node(customer_id, kind="customer", provenance="source customer record")
    for event in events:
        event_id = event["event_id"]
        graph.add_node(event_id, kind="event", event_type=event["event_type"], timestamp=event["timestamp"],
                       provenance="simulated")
        graph.add_edge(customer_id, event_id, relation="has_event")
        indicator = event.get("indicator")
        if indicator:
            node = f"indicator:{indicator}"
            graph.add_node(node, kind="indicator", provenance="derived from aggregate feature")
            graph.add_edge(event_id, node, relation="illustrates")
    return {"nodes": [{"id": n, **d} for n, d in graph.nodes(data=True)],
            "edges": [{"source": a, "target": b, **d} for a, b, d in graph.edges(data=True)],
            "provenance": "Customer aggregate fields are source data; timestamped events are simulated."}
