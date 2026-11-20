"""AURA Streamlit workspace. All customer and workflow data comes from FastAPI."""
from datetime import datetime
from pathlib import Path
from urllib.parse import quote
import json

import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from frontend.api import AuraAPI, APIError
    from frontend.components import apply_theme, page_header, status_badge, chart_layout
except ModuleNotFoundError:
    from api import AuraAPI, APIError
    from components import apply_theme, page_header, status_badge, chart_layout


APP_TITLE = "AURA — AI-Powered User Retention Analytics"
ROOT = Path(__file__).resolve().parents[1]
MODEL_REPORTS = ROOT / "ml" / "reports"
st.set_page_config(page_title=APP_TITLE, page_icon="◈", layout="wide", initial_sidebar_state="expanded")
apply_theme()


def api_client():
    return AuraAPI(st.session_state.get("access_token"))


def call(method, path, *, params=None, payload=None, files=None, timeout=12, raw=False, quiet=False):
    try:
        return api_client().request(method, path, params=params, json=payload, files=files, timeout=timeout, raw=raw)
    except APIError as exc:
        if exc.status_code == 409:
            detail=(exc.payload or {}).get("detail",{})
            if isinstance(detail,dict) and detail.get("tenants"):
                st.session_state.login_tenants=detail["tenants"]
                st.warning("This account belongs to multiple organizations. Select one and sign in again.")
                st.rerun()
        if exc.status_code == 401 and st.session_state.get("access_token"):
            for key in ("access_token", "user", "tenant"): st.session_state.pop(key, None)
            st.warning("Your session expired. Please sign in again.")
            st.stop()
        if not quiet: st.error(str(exc))
        return None


def get(path, *, quiet=False, **params): return call("GET", path, params=params or None, quiet=quiet)
def post(path, payload=None, timeout=15): return call("POST", path, payload=payload, timeout=timeout)
def put(path, payload): return call("PUT", path, payload=payload)


def show_frame(data, *, empty="No records to display.", height=None):
    frame = pd.DataFrame(data or [])
    if frame.empty:
        st.info(empty)
        return frame
    options={"width":"stretch","hide_index":True}
    if height is not None: options["height"]=height
    st.dataframe(frame, **options)
    return frame


def chart_empty():
    st.caption("No stored records are available for this chart yet.")


def login_view():
    st.markdown('<div class="aura-eyebrow">CUSTOMER RETENTION INTELLIGENCE</div>', unsafe_allow_html=True)
    st.title("AURA")
    st.markdown("### AI-Powered User Retention Analytics")
    st.markdown("Explore customer risk, investigate supporting evidence, and review retention recommendations through a controlled workflow.")
    left, right = st.columns([1.15, .85], gap="large")
    with left:
        st.markdown("#### A clear path from signal to review")
        st.markdown("Detect customer signals · Understand evidence · Estimate risk · Rank options · Require human approval")
        st.caption("The demonstration dataset and event timeline are synthetic. Interventions are simulated; outcomes remain unknown until separately observed.")
    with right:
        mode = st.radio("Account", ["Sign in", "Create organization"], horizontal=True)
        with st.form("auth_form", border=True):
            email = st.text_input("Work email", autocomplete="email")
            password = st.text_input("Password", type="password", autocomplete="current-password")
            tenants=st.session_state.get("login_tenants",[])
            selected_tenant=(st.selectbox("Organization",tenants,format_func=lambda item:item["name"]) if mode=="Sign in" and tenants else None)
            if mode == "Create organization":
                full_name = st.text_input("Your name")
                organization = st.text_input("Organization")
                industry = st.text_input("Industry (optional)")
            submit = st.form_submit_button("Sign in" if mode == "Sign in" else "Create organization", type="primary", width="stretch")
        if submit:
            if not email.strip() or not password:
                st.error("Enter your email and password.")
            else:
                payload = {"email":email.strip(),"password":password}
                if selected_tenant: payload["tenant_id"]=selected_tenant["id"]
                if mode == "Create organization":
                    if not full_name.strip() or not organization.strip():
                        st.error("Your name and organization are required.")
                        return
                    payload.update({"full_name":full_name.strip(),"organization_name":organization.strip(),"industry":industry.strip() or None})
                result = call("POST", "/api/auth/login" if mode == "Sign in" else "/api/auth/signup", payload=payload)
                if result and result.get("access_token"):
                    st.session_state.pop("login_tenants",None)
                    st.session_state.access_token = result["access_token"]
                    st.session_state.user = result["user"]
                    st.session_state.tenant = result["tenant"]
                    st.rerun()


def overview_page(context, roles):
    page_header("Overview", "A tenant-scoped view of model predictions, analysis, and recommendations.")
    with st.spinner("Loading workspace metrics…"):
        data = get("/api/dashboard/overview")
        datasets = get("/api/datasets")
        threshold = get("/api/reports/threshold")
        report_summary = get("/api/reports/overview")
    if data is None: return
    risk = data["prediction_risk_distribution"]
    unavailable = "Unavailable"
    metrics=[("Total customers",data.get("total_customers", unavailable)),
        ("High / critical risk",risk.get("HIGH",0)+risk.get("CRITICAL",0)),
        ("Medium risk",risk.get("MEDIUM",0)),("Low risk",risk.get("LOW",0)),
        ("Scored customers",data["analyzed_customers"]),
        ("Pending review",data["pending_recommendations"]),
        ("Approved actions",data.get("approved_interventions",0)),
        ("Analysis runs",report_summary.get("analysis_runs", unavailable) if report_summary is not None else unavailable),
        ("Simulations · 7d",data.get("recent_simulated_executions_7d",0))]
    for start in range(0,len(metrics),3):
        cols=st.columns(3)
        for col,(label,value) in zip(cols,metrics[start:start+3]):
            col.metric(label, f"{value:,}" if isinstance(value, (int, float)) else str(value))

    labels_data = data.get("dataset_label_summary") or {}
    versions = datasets.get("items", []) if datasets is not None else []
    latest_dataset = versions[0] if versions else None
    model_metadata_path=ROOT/"ml"/"artifacts"/"model_metadata.json"
    model_metadata=json.loads(model_metadata_path.read_text()) if model_metadata_path.exists() else {}
    model_version = model_metadata.get("model_version","unknown")
    first, second = st.columns([1.25,.75])
    with first:
        st.markdown("#### Workspace status")
        st.markdown(f"**Welcome, {context['user']['full_name']}** · {datetime.now().strftime('%A, %d %B %Y')}")
        st.markdown(f"**Organization:** {context['tenant']['name']}")
        dataset_status = latest_dataset['version'] if latest_dataset else ('Unavailable' if datasets is None else 'No dataset imported')
        st.markdown(f"**Dataset:** {dataset_status}")
        st.markdown(f"**Active model:** {model_version} · threshold {(threshold or {}).get('backend_active_threshold','—')}")
        st.caption("Risk totals include customers with a stored prediction. Dataset labels are historical synthetic labels and are not new predictions.")
    with second:
        if not data["total_customers"] and roles.intersection({"OWNER","ADMIN"}):
            st.info("Import the supplied data to start the workspace.")
            if st.button("Load supplied 12,000-row demo dataset", type="primary"):
                with st.spinner("Importing supplied demonstration data…"):
                    result = post("/api/datasets/demo", timeout=90)
                if result: st.success(f"Imported {result['row_count']:,} source records."); st.rerun()
        elif not data["total_customers"]:
            st.info("Ask an organization owner or administrator to import customer data.")
        if labels_data.get("target_present"):
            st.caption(f"Synthetic labels in latest dataset: {labels_data.get('churned_labels',0):,} churned · {labels_data.get('row_count',0)-labels_data.get('churned_labels',0):,} non-churned.")

    st.divider()
    left, right = st.columns(2)
    with left:
        st.markdown("#### Stored prediction distribution")
        risk_df = pd.DataFrame([{"Risk":k.title(),"Customers":v} for k,v in risk.items() if v])
        if not risk_df.empty:
            fig = px.pie(risk_df, names="Risk", values="Customers", hole=.58,
                color="Risk", color_discrete_map={"Critical":"#b91c1c","High":"#ea580c","Medium":"#d97706","Low":"#15803d"})
            st.plotly_chart(chart_layout(fig, 310), width="stretch")
        else: chart_empty()
        st.caption("Predictions are model estimates. Customers without stored predictions are excluded.")
    with right:
        st.markdown("#### Customer segments")
        seg = pd.DataFrame(data.get("segment_distribution",[]))
        if not seg.empty:
            fig = px.bar(seg, x="customers", y="segment", orientation="h", labels={"customers":"Customers","segment":"Segment"}, text="customers")
            fig.update_traces(marker_color="#2563eb", textposition="outside")
            st.plotly_chart(chart_layout(fig, 310), width="stretch")
        else: chart_empty()
    if labels_data.get("target_present") and labels_data.get("churned_labels") is not None:
        churned=int(labels_data.get("churned_labels",0)); total=int(labels_data.get("row_count",0))
        st.markdown("#### Historical labels in latest imported dataset")
        label_frame=pd.DataFrame([{"Dataset label":"Churn label","Customers":churned},
                                  {"Dataset label":"Non-churn label","Customers":max(0,total-churned)}])
        fig=px.pie(label_frame,names="Dataset label",values="Customers",hole=.55,
            color="Dataset label",color_discrete_map={"Churn label":"#ea580c","Non-churn label":"#94a3b8"})
        st.plotly_chart(chart_layout(fig,280),width="stretch")
        st.caption("Historical target labels from the uploaded synthetic dataset. They are not current model predictions.")
    left, right = st.columns(2)
    with left:
        st.markdown("#### Recommendation status")
        status = pd.DataFrame([{"Status":k.replace("_"," ").title(),"Count":v} for k,v in data.get("recommendation_status_distribution",{}).items()])
        if not status.empty:
            fig = px.bar(status, x="Status", y="Count", color="Status", color_discrete_sequence=["#2563eb"])
            st.plotly_chart(chart_layout(fig, 300), width="stretch")
        else: chart_empty()
    with right:
        st.markdown("#### Recent analysis runs")
        rows = data.get("recent_analysis_runs",[])
        if rows:
            for row in rows:
                a,b,c = st.columns([1.25,1.1,1])
                a.caption((row.get("customer_id") or "Customer")+" · "+row["created_at"][:16].replace("T"," "))
                b.write(row["status"].replace("_"," ").title())
                c.button("Inspect",key=f"dashboard_run_{row['run_id']}",
                    on_click=_open_analysis_run,args=(row["run_id"],row.get("customer_id")))
        else: st.info("No analysis runs yet. Open a customer profile to start one.")
    recent_recs=get("/api/recommendations",limit=5)
    recent_agents=[]
    for run_item in data.get("recent_analysis_runs",[])[:3]:
        run_detail=get(f"/api/agents/runs/{run_item['run_id']}",quiet=True)
        if run_detail:
            recent_agents.extend([{**agent,"analysis_run_id":run_item["run_id"]} for agent in run_detail.get("agents",[])])
    t_recs,t_agents=st.tabs(["Recent recommendations","Recent agent activity"])
    with t_recs:
        rec_items=(recent_recs or {}).get("items",[])
        if rec_items:
            for rec in rec_items:
                st.markdown(f"`{rec['customer_id']}` · **{rec['action_name']}** · {rec['status'].replace('_',' ').title()} · MCIA {rec['mcia']['mcia_score']:.3f}")
        else: st.info("No recommendation records yet.")
    with t_agents:
        recent_agents=sorted(recent_agents,key=lambda x:x.get("started_at") or "",reverse=True)[:8]
        show_frame([{"Agent":x["name"],"Status":x["status"],"Run":x["analysis_run_id"][:8],"Started":(x.get("started_at") or "")[:16]} for x in recent_agents],empty="No agent activity yet.")
    models = get("/api/reports/models")
    if models:
        with st.expander("Held-out model comparison (all candidates at 0.50)"):
            show_frame(models)


def customers_page():
    page_header("Customers", "Search and filter tenant records, then open a profile for prediction and analysis.")
    dashboard = get("/api/dashboard/overview")
    segments = [x["segment"] for x in (dashboard or {}).get("segment_distribution",[])]
    with st.form("customer_filters"):
        c1,c2,c3,c4 = st.columns([1.4,1,1,1])
        search = c1.text_input("Customer ID")
        risk = c2.selectbox("Risk", ["All","Critical","High","Medium","Low","Not analyzed"])
        segment = c3.selectbox("Segment", ["All", *segments])
        sort_label = c4.selectbox("Sort by", ["Customer ID","Segment","Risk","Churn probability"])
        c1,c2,c3 = st.columns([1,1,2])
        per_page = c1.selectbox("Rows per page", [25,50,100,200], index=1)
        page_number = c2.number_input("Page", min_value=1, value=1, step=1)
        descending = c3.toggle("Descending", value=False)
        apply_filter = st.form_submit_button("Apply filters", type="primary")
    if apply_filter or "customer_query" not in st.session_state:
        st.session_state.customer_query = {"search":search,"risk":risk,"segment":segment,"sort":sort_label,
            "page":int(page_number),"size":per_page,"descending":descending}
    query = st.session_state.customer_query
    sort_map={"Customer ID":"customer_id","Segment":"segment","Risk":"risk_level","Churn probability":"churn_probability"}
    params={"limit":query["size"],"offset":(query["page"]-1)*query["size"],"sort_by":sort_map[query["sort"]],"descending":query["descending"]}
    if query["search"]: params["search"]=query["search"]
    if query["risk"]!="All": params["risk_level"]=query["risk"].upper().replace(" ","_")
    if query["segment"]!="All": params["segment"]=query["segment"]
    with st.spinner("Loading customers…"):
        data=get("/api/customers",**params)
    if not data: return
    st.caption(f"{data['total']:,} matching records · page {query['page']} · {data['limit']} rows per page")
    frame=pd.DataFrame(data["items"])
    if not frame.empty:
        # Keep the main grid legible at a laptop viewport; detailed source fields
        # remain available on the customer profile.
        preferred=["customer_id","customer_segment","tenure_months","risk_level","churn_probability"]
        visible=[column for column in preferred if column in frame]
        customer_view=frame[visible].copy()
        if "churn_probability" in customer_view:
            customer_view["churn_probability"]=customer_view["churn_probability"]*100
        st.dataframe(customer_view,width="stretch",hide_index=True,
            column_config={"churn_probability":st.column_config.NumberColumn("Churn probability (%)",format="%.1f")})
    else: st.info("No customers match these filters.")
    if not frame.empty:
        selected=st.selectbox("Open profile",frame["customer_id"].tolist(),key="customer_open_id")
        st.button("Open customer profile",type="primary",on_click=_open_customer_profile,args=(selected,))


def _open_customer_profile(customer_id):
    st.session_state.selected_customer_id=customer_id
    st.session_state.active_page="Customer profile"


def _open_analysis_run(run_id, customer_id):
    st.session_state.selected_analysis_run=run_id
    st.session_state.selected_customer_id=customer_id
    st.session_state.active_page="Customer profile"


def graph_figure(graph):
    network=nx.Graph()
    for node in graph.get("nodes",[]): network.add_node(node["id"],**node)
    for edge in graph.get("edges",[]): network.add_edge(edge["source"],edge["target"],**edge)
    if not network.number_of_nodes(): return None
    pos=nx.spring_layout(network,seed=17)
    ex=[];ey=[]
    for a,b in network.edges:
        ex += [pos[a][0],pos[b][0],None]; ey += [pos[a][1],pos[b][1],None]
    edge=go.Scatter(x=ex,y=ey,mode="lines",line={"width":1,"color":"#cbd5e1"},hoverinfo="none")
    nodes=list(network.nodes)
    labels=[network.nodes[n].get("event_type",network.nodes[n].get("id",n)) for n in nodes]
    colors=["#1d4ed8" if network.nodes[n].get("kind")=="customer" else "#0f766e" if network.nodes[n].get("kind")=="event" else "#b45309" for n in nodes]
    hover=[network.nodes[n].get("provenance","") for n in nodes]
    node=go.Scatter(x=[pos[n][0] for n in nodes],y=[pos[n][1] for n in nodes],mode="markers+text",
        text=labels,textposition="top center",customdata=hover,hovertemplate="%{text}<br>%{customdata}<extra></extra>",
        marker={"size":18,"color":colors,"line":{"color":"white","width":1}})
    fig=go.Figure([edge,node]); fig.update_layout(showlegend=False,xaxis={"visible":False},yaxis={"visible":False})
    return chart_layout(fig,460)


def agent_view(run):
    records=run.get("agent_executions",[])
    ordered=["Detect","Diagnose","Predict","Strategy","HumanApproval","Execute"]
    st.markdown("#### Workflow progression")
    lookup={}
    for item in records:
        base=item["agent"].split(":",1)[0]
        lookup.setdefault(base,[]).append(item)
    cols=st.columns(len(ordered))
    for col,name in zip(cols,ordered):
        matching=lookup.get(name,[])
        state=matching[-1]["status"] if matching else ("awaiting approval" if name=="HumanApproval" and run.get("status")=="pending_approval" else "pending")
        col.markdown(f"**{name}**")
        col.caption(state.replace("_"," ").title())
    for item in records:
        with st.expander(f"{item['agent']} · {item['status'].replace('_',' ').title()}"):
            st.caption(f"Started: {item.get('started_at','—')} · Completed: {item.get('completed_at') or '—'}")
            if item.get("error"): st.error(item["error"])
            output=item.get("output")
            if output:
                if item["agent"].startswith("Diagnose"):
                    st.caption(f"Output provenance: {output.get('provenance','unspecified')}")
                elif item["agent"].startswith("Strategy"):
                    st.caption("Per-action rationale provenance is shown in recommendation details.")
                st.json(output)


def customer_profile_page(roles):
    page_header("Customer profile", "Inspect stored features, prediction history, simulated events, and workflow outcomes.")
    selected=st.session_state.get("selected_customer_id","")
    customer_id=st.text_input("Customer ID",value=selected,key="profile_customer_id")
    if not customer_id.strip():
        st.info("Choose a customer from the Customers page or enter a customer ID.")
        return
    with st.spinner("Loading customer profile…"):
        profile=get(f"/api/customers/{quote(customer_id.strip(),safe='')}")
    if not profile: return
    st.session_state.selected_customer_id=customer_id.strip()
    pred=profile.get("prediction")
    st.markdown(f"### {profile['customer_id']}")
    if pred:
        m=st.columns(4)
        m[0].metric("Churn probability",f"{pred['churn_probability']:.1%}")
        m[1].metric("Risk category",pred["risk_level"].title())
        m[2].metric("Decision threshold",f"{pred['decision_threshold']:.2f}")
        m[3].metric("Model version",pred["model_version"])
        st.caption(f"Prediction recorded {pred['prediction_timestamp']} · a model estimate, not a guarantee.")
    else:
        st.info("No prediction has been stored for this customer.")
        if st.button("Run saved model prediction"):
            features={k:v for k,v in profile.items() if k not in {"customer_id","prediction"}}
            result=post("/api/predictions",{"customer_id":profile["customer_id"],"features":features})
            if result: st.success("Prediction stored."); st.rerun()

    with st.expander("Customer attributes",expanded=False):
        attr={k:v for k,v in profile.items() if k not in {"customer_id","prediction"}}
        # A single display column may contain both numeric and categorical source
        # values; normalize it for Arrow/Streamlit's homogeneous column encoding.
        show_frame([{"Attribute":k,"Value":str(v)} for k,v in attr.items()])
    events=get(f"/api/customers/{customer_id}/events")
    if events:
        st.markdown("#### Event timeline")
        st.caption(events["provenance"])
        event_frame=pd.DataFrame(events.get("items",[]))
        if not event_frame.empty:
            event_frame["timestamp"]=pd.to_datetime(event_frame["timestamp"],errors="coerce",utc=True)
            event_frame=event_frame.sort_values("timestamp")
            event_frame=event_frame[[column for column in ("event_type","timestamp","severity") if column in event_frame]]
            st.dataframe(event_frame,width="stretch",hide_index=True)
    st.markdown("#### Risk analysis")
    col1,col2=st.columns([1,1])
    with col1:
        if st.button("Run five-agent analysis",type="primary",disabled=not bool(roles.intersection({"ANALYST","APPROVER","OWNER","ADMIN"}))):
            with st.spinner("Running Detect → Diagnose → Predict → Strategy…"):
                run=post(f"/api/customers/{customer_id}/analyze",timeout=60)
            if run:
                st.session_state.selected_analysis_run=run["run_id"]
                st.success(f"Workflow paused for review: {run['run_id']}")
                st.rerun()
    with col2:
        runs=get("/api/analysis",limit=100)
        customer_runs=[row for row in (runs or {}).get("items",[]) if row["customer_id"]==customer_id]
        run_ids=[row["run_id"] for row in customer_runs]
        default=st.session_state.get("selected_analysis_run")
        if run_ids:
            chosen=st.selectbox("Analysis run",run_ids,index=run_ids.index(default) if default in run_ids else 0)
        else: chosen=None; st.caption("No analysis history for this customer yet.")
    if chosen:
        st.session_state.selected_analysis_run=chosen
        run=get(f"/api/analysis/{chosen}")
        if run:
            status=run["status"].replace("_"," ").title()
            st.markdown(f"**Workflow status:** {status} · **Run ID:** `{chosen}`")
            agent_view(run)
            result=run.get("result") or {}
            t1,t2,t3=st.tabs(["Signals & diagnosis","Recommendations","Event graph"])
            with t1:
                detect=result.get("detect")
                diagnose=result.get("diagnose")
                if detect: show_frame(detect.get("signals",[]),empty="No risk signals were identified.")
                if diagnose:
                    st.caption(f"Diagnosis provenance: {diagnose.get('provenance','not specified')}")
                    st.caption(diagnose.get("summary",""))
                    show_frame(diagnose.get("items",[]),empty="No evidence-backed hypotheses were returned.")
            with t2:
                recommendations=get("/api/recommendations",limit=100,customer_id=customer_id)
                recs=[r for r in (recommendations or {}).get("items",[]) if r.get("analysis_run_id")==chosen]
                if recs:
                    for rec in recs:
                        with st.expander(f"{rec['action_name']} · {rec['status'].replace('_',' ').title()}"):
                            st.write(rec["rationale"]); st.json({"eligibility":rec["eligibility"],"MCIA":rec["mcia"]})
                            st.caption("Rationale source: " + next((x.get("rationale_provenance","deterministic MCIA and eligibility rules") for x in result.get("strategy",[]) if x.get("recommendation_id")==rec["id"]),"recorded rule-based explanation"))
                else: st.info("No recommendation records for this run.")
            with t3:
                graph=get(f"/api/customers/{customer_id}/graph")
                if graph:
                    fig=graph_figure(graph)
                    if fig: st.plotly_chart(fig,width="stretch")
                    st.caption(graph["provenance"]+" Edges are descriptive, not causal.")


def model_page():
    page_header("Model report", "Model comparison, threshold trade-offs, and evaluation limits.")
    st.warning("All reported results use a synthetic dataset. They do not validate performance on real customers.")
    comparison=get("/api/reports/models")
    threshold=get("/api/reports/threshold")
    metadata=json.loads((ROOT/"ml"/"artifacts"/"model_metadata.json").read_text()) if (ROOT/"ml"/"artifacts"/"model_metadata.json").exists() else {}
    if comparison:
        st.markdown("#### Candidate comparison · held-out split, fixed 0.50 threshold")
        table=pd.DataFrame(comparison)
        table=table[[column for column in ("model","accuracy","precision","recall","f1","roc_auc","pr_auc") if column in table]]
        table=table.rename(columns={"model":"Model","accuracy":"Accuracy","precision":"Precision","recall":"Recall","f1":"F1","roc_auc":"ROC-AUC","pr_auc":"PR-AUC"})
        st.dataframe(table,width="stretch",hide_index=True)
        plot_data=table.melt(id_vars="Model",value_vars=["Precision","Recall","F1","ROC-AUC","PR-AUC"],var_name="Metric",value_name="Score")
        fig=px.bar(plot_data,x="Metric",y="Score",color="Model",barmode="group",range_y=[0,1])
        st.plotly_chart(chart_layout(fig),width="stretch")
    if threshold:
        result=threshold["held_out_test_metrics"]
        cols=st.columns(5)
        cols[0].metric("Selected model",metadata.get("selected_model","XGBoost").upper())
        cols[1].metric("Model version",metadata.get("model_version","—"))
        cols[2].metric("Active threshold",f"{threshold['backend_active_threshold']:.2f}")
        cols[3].metric("Outreach capacity",f"{threshold['capacity_rate']:.0%}")
        cols[4].metric("Input features",len(metadata.get("features",[])))
        st.caption(f"Training completed (UTC): {metadata.get('trained_at_utc','not recorded')} · model selection rule: {metadata.get('selection_rule','not recorded')}")
        st.markdown("#### Operational threshold · held-out test once")
        st.caption(threshold["selection_rationale"])
        metrics={k:result[k] for k in ("precision","recall","f1","specificity","false_positive_rate","false_negative_rate","flagged_rate","brier_score") if k in result}
        cols=st.columns(4)
        for col,(key,value) in zip(cols,metrics.items()): col.metric(key.replace("_"," ").title(),f"{value:.3f}" if value is not None else "—")
        cm=result["confusion_matrix"]
        st.markdown("**Confusion matrix** · rows = actual class [non-churn, churn], columns = predicted [not flagged, flagged]")
        st.dataframe(pd.DataFrame(cm,index=["Actual non-churn","Actual churn"],columns=["Predicted not flagged","Predicted flagged"]),width="stretch")
        st.markdown("#### Ranking metrics at contact capacity")
        show_frame(result.get("ranking_metrics",[]))
        plot_cols=st.columns(2)
        for col,name,label in ((plot_cols[0],"precision_recall_comparison.png","Precision-recall curves"),(plot_cols[1],"threshold_performance.png","Threshold performance")):
            path=MODEL_REPORTS/name
            if path.exists(): col.image(str(path),caption=label,width="stretch")
        importance=MODEL_REPORTS/"feature_importance.png"
        if importance.exists(): st.image(str(importance),caption="XGBoost feature importance · predictive association, not causal attribution",width="stretch")
        with st.expander("Evaluation limitations"):
            st.markdown("- Operational threshold was selected from training-only out-of-fold probabilities; the held-out split was not used for selection.\n- Recall at the selected threshold remains limited, so some churn labels are missed.\n- The 20% capacity is an explicit planning assumption.\n- The Brier score is a diagnostic; no calibration transform is applied.\n- Model probabilities are not guarantees, treatment effects, or proof of retention.")


def causal_page():
    page_header("Causal explorer", "Explore event and feature relationships as hypotheses, with provenance visible.")
    st.info("The supplied aggregate snapshot has no defensible intervention assignment and longitudinal outcome design. AURA reports no causal effect estimate.")
    data=get("/api/customers",limit=100,offset=0,sort_by="customer_id")
    if not data or not data["items"]: st.info("Import customer data to explore a customer graph."); return
    customer_id=st.selectbox("Customer",[r["customer_id"] for r in data["items"]],key="causal_customer")
    if st.button("Load descriptive graph"):
        graph=get(f"/api/customers/{customer_id}/graph")
        if graph: st.session_state.causal_graph=graph
    graph=st.session_state.get("causal_graph")
    if graph:
        fig=graph_figure(graph)
        if fig: st.plotly_chart(fig,width="stretch")
        st.caption(graph["provenance"])
        st.markdown("**Legend:** blue = customer · teal = simulated event · amber = aggregate-feature indicator")
        left,right=st.columns(2)
        with left: st.markdown("#### Event sequence"); show_frame([n for n in graph["nodes"] if n["kind"]=="event"],empty="No event nodes.")
        with right: st.markdown("#### Relationships"); show_frame(graph["edges"],empty="No graph edges.")
        st.caption("Edges describe how the simulator linked nodes. They do not establish cause.")


def actions_page(roles):
    page_header("Retention center", "Review proposed actions, evidence, eligibility, and simulated execution status.")
    tab_pending,tab_all=st.tabs(["Pending review","All recommendations"])
    with tab_pending:
        pending=get("/api/recommendations",limit=100,status="pending_approval")
        if not pending or not pending.get("items"):
            st.info("No recommendations are waiting for review.")
        else:
            st.caption(f"{pending['total']} recommendations awaiting an authorized reviewer.")
            for rec in pending["items"]:
                with st.container(border=True):
                    c1,c2,c3=st.columns([2,1,1])
                    c1.markdown(f"**{rec['action_name']}** · Customer `{rec['customer_id']}`")
                    c2.metric("MCIA score",f"{rec['mcia']['mcia_score']:.3f}")
                    c3.markdown("Status"); status_badge(rec["status"])
                    st.write(rec["rationale"])
                    risk_label=rec.get("risk_level","NOT_ANALYZED").replace("_"," ").title()
                    st.caption(f"Risk {risk_label} · Probability {rec['churn_probability']:.1%}" if rec.get("churn_probability") is not None else f"Risk {risk_label}")
                    st.caption(f"Eligibility: {rec['eligibility']} · Score is relative decision support, not churn probability or ROI.")
                    with st.expander("Review evidence and score"):
                        detail=get(f"/api/recommendations/{rec['id']}")
                        st.json({"M/T/R/F":rec["mcia"],"approval":(detail or {}).get("approval"),"execution":(detail or {}).get("execution")})
                        run=get(f"/api/analysis/{rec['analysis_run_id']}") if rec.get("analysis_run_id") else None
                        if run:
                            result=run.get("result") or {}
                            st.caption("Prediction and risk signals for this recommendation")
                            st.json({"prediction":result.get("prediction"),"signals":(result.get("detect") or {}).get("signals"),
                                     "diagnosis":(result.get("diagnose") or {}).get("items")})
                    if roles.intersection({"APPROVER","OWNER","ADMIN"}):
                        with st.form(f"review_{rec['id']}"):
                            note=st.text_input("Review note (optional)",key=f"note_{rec['id']}")
                            confirm=st.checkbox("I reviewed this simulated recommendation",key=f"confirm_{rec['id']}")
                            a,b=st.columns(2)
                            # Streamlit form controls do not rerun on checkbox changes,
                            # so disabling these buttons from the checkbox state traps
                            # reviewers. Keep them submit-capable and enforce confirmation
                            # after the form submits instead.
                            approve=a.form_submit_button("Approve and simulate",type="primary")
                            reject=b.form_submit_button("Reject")
                        if approve or reject:
                            if not confirm:
                                st.warning("Confirm that you reviewed this simulated recommendation before submitting.")
                            else:
                                decision="approve" if approve else "reject"
                                answer=post(f"/api/recommendations/{rec['id']}/{decision}",{"note":note or None})
                                if answer: st.success(answer["message"]); st.rerun()
                    else: st.caption("Your current role does not permit approval or rejection.")
    with tab_all:
        status=st.selectbox("Filter by status",["All","pending_approval","executed","rejected","execution_failed","approved"])
        params={"limit":100}
        if status!="All": params["status"]=status
        recs=get("/api/recommendations",**params)
        show_frame((recs or {}).get("items",[]),empty="No recommendations in this status.")


def history_page():
    page_header("History & activity", "Tenant-scoped analysis, agents, decisions, executions, outcomes, and audit records.")
    if st.button("Refresh activity"): st.rerun()
    t_runs,t_agents,t_rec,t_exec,t_audit=st.tabs(["Analysis runs","Agent executions","Recommendations","Simulations & outcomes","Audit activity"])
    with t_runs:
        limit=st.selectbox("Rows per page",[25,50,100,200],key="history_limit")
        page=st.number_input("Page",min_value=1,value=1,key="history_page")
        runs=get("/api/analysis",limit=limit,offset=(page-1)*limit)
        run_items=(runs or {}).get("items",[])
        run_view=[{"run_id":row.get("run_id","")[:8],"customer_id":row.get("customer_id"),
            "status":row.get("status"),"created_at":row.get("created_at")} for row in run_items]
        frame=show_frame(run_view,empty="No analysis runs are stored.")
        if not frame.empty:
            run_id=st.selectbox("Inspect run",[row["run_id"] for row in run_items])
            detail=get(f"/api/analysis/{run_id}")
            if detail:
                st.markdown(f"Workflow status: **{detail['status']}**")
                agent_view(detail)
    with t_agents:
        runs=get("/api/analysis",limit=100)
        run_ids=[x["run_id"] for x in (runs or {}).get("items",[])]
        if run_ids:
            run_id=st.selectbox("Run",run_ids,key="agent_run_select")
            agents=get(f"/api/agents/runs/{run_id}")
            if agents:
                st.caption(f"Workflow status: {agents['run_status']}")
                agent_rows=[{key:row.get(key) for key in ("agent_name","status","started_at","completed_at","error")}
                             for row in agents.get("agents",[])]
                show_frame(agent_rows,empty="No agent records.")
        else: st.info("No agent executions yet.")
    with t_rec:
        status=st.selectbox("Recommendation status",["All","pending_approval","executed","rejected","execution_failed"],key="hist_rec_status")
        params={"limit":100}
        if status!="All": params["status"]=status
        recs=get("/api/recommendations",**params)
        recommendation_view=[{"id":row.get("id","")[:8],"customer_id":row.get("customer_id"),
            "action_name":row.get("action_name"),"status":row.get("status"),"risk_level":row.get("risk_level"),
            "churn_probability":round(row["churn_probability"]*100,1) if row.get("churn_probability") is not None else None}
            for row in (recs or {}).get("items",[])]
        show_frame(recommendation_view,empty="No recommendations found.")
    with t_exec:
        interventions=get("/api/interventions",limit=100)
        items=(interventions or {}).get("items",[])
        execution_view=[{"execution_id":row.get("execution_id","")[:8],"customer_id":row.get("customer_id"),
            "action_name":row.get("action_name"),"status":row.get("status"),"simulated":row.get("simulated"),
            "executed_at":row.get("executed_at")} for row in items]
        show_frame(execution_view,empty="No simulated executions are stored.")
        if items:
            execution=st.selectbox("Record follow-up for execution",items,format_func=lambda x:f"{x['action_name']} · {x['customer_id']} · {x['execution_id'][:8]}")
            with st.form("outcome_form"):
                outcome=st.selectbox("Manually observed follow-up",["unknown","retained","churned","no_response"])
                response=st.text_input("Customer response (optional)")
                notes=st.text_area("Notes (optional)")
                submit=st.form_submit_button("Record follow-up")
            if submit:
                result=post(f"/api/interventions/{execution['execution_id']}/outcome",{"outcome_status":outcome,"customer_response":response or None,"notes":notes or None})
                if result: st.success("Follow-up recorded. It is not a causal treatment-effect estimate."); st.rerun()
    with t_audit:
        audit=get("/api/audit",limit=100)
        audit_view=[{"action":row.get("action"),"resource_type":row.get("resource_type"),
            "resource_id":(row.get("resource_id") or "")[:8],"created_at":row.get("created_at")}
            for row in (audit or {}).get("items",[])]
        show_frame(audit_view,empty="No audit events are recorded.")


def datasets_page(roles):
    page_header("Datasets", "Manage tenant customer records without changing the trained model.")
    data=get("/api/datasets")
    if data:
        st.markdown("#### Imported versions")
        versions=[{"Version":row.get("version"),"Filename":row.get("filename"),
            "Rows":row.get("row_count"),"Imported":row.get("created_at")} for row in data.get("items",[])]
        show_frame(versions,empty="No customer dataset has been imported.")
    if "OWNER" in roles or "ADMIN" in roles:
        st.markdown("#### Import customer CSV")
        st.caption("Required schema: customer_id plus all 34 dictionary input features. Optional churn labels are validation metadata only. Importing never retrains the active model.")
        upload=st.file_uploader("Choose a CSV file",type=["csv"],key="dataset_upload")
        if upload:
            left,right=st.columns(2)
            if left.button("Validate and preview",type="primary"):
                result=call("POST","/api/datasets/validate",files={"file":(upload.name,upload.getvalue(),"text/csv")},timeout=45)
                if result:
                    st.session_state.dataset_validation=result
                    st.session_state.dataset_file_name=upload.name
            validation=st.session_state.get("dataset_validation")
            if validation and st.session_state.get("dataset_file_name")==upload.name:
                report=validation["validation"]
                if report["valid"]: st.success(f"Validation passed · {report['row_count']:,} rows · {report['column_count']} columns")
                else: st.error("Validation failed. Correct the reported errors before import.")
                st.json(report)
                show_frame(validation.get("preview",[]),empty="No preview rows.")
                if report["valid"] and right.button("Import validated CSV"):
                    result=call("POST","/api/datasets/upload",files={"file":(upload.name,upload.getvalue(),"text/csv")},timeout=90)
                    if result:
                        st.success(f"Imported {result['row_count']:,} customer records. The active model was not changed.")
                        st.session_state.pop("dataset_validation",None); st.rerun()
    else: st.info("Dataset import is limited to organization owners and administrators.")


def settings_page(context, roles):
    page_header("Settings", "Organization, access, model, MCIA, and optional LLM controls.")
    profile=get("/api/auth/me")
    settings=get("/api/settings")
    tabs=st.tabs(["Profile & organization","Model","MCIA","LLM","Users"])
    with tabs[0]:
        st.markdown("#### Signed-in profile")
        user=(profile or {}).get("user",{})
        st.markdown(f"**Name:** {user.get('full_name','—')}  ·  **Email:** {user.get('email','—')}")
        st.markdown("#### Organization")
        tenant=(settings or {}).get("tenant",context.get("tenant",{}))
        st.markdown(f"**Name:** {tenant.get('name','—')}  ·  **Industry:** {tenant.get('industry') or 'Not specified'}")
        st.caption("Organization details are read-only in this version.")
    with tabs[1]:
        threshold=get("/api/reports/threshold")
        if threshold:
            st.metric("Prediction threshold",f"{threshold['backend_active_threshold']:.2f}")
            st.caption("Set the active threshold with AURA_PREDICTION_THRESHOLD on the backend. Outreach capacity is selected during training; changing this server value does not retrain the model.")
            st.caption(f"Source: {threshold['active_threshold_source']} · model: {threshold['selected_model']}")
    with tabs[2]:
        if not roles.intersection({"OWNER","ADMIN"}): st.info("MCIA settings are limited to owners and administrators.")
        elif settings:
            current=settings["mcia_weights"]
            with st.form("mcia_form"):
                weights={key:st.number_input(f"{key.title()} weight",min_value=0.0,max_value=10.0,value=float(current[key]),step=.1) for key in ("money","time","relation","feasibility")}
                st.caption("Weights are normalized on calculation. Money and Time are costs; Relationship and Feasibility are benefits.")
                save=st.form_submit_button("Save MCIA weights",type="primary")
            if save:
                result=put("/api/settings/mcia",weights)
                if result: st.success("Weights saved for future recommendations."); st.rerun()
    with tabs[3]:
        if not roles.intersection({"OWNER","ADMIN"}): st.info("LLM authorization is limited to owners and administrators.")
        else:
            status=get("/api/settings/llm")
            if status:
                cols=st.columns(4)
                cols[0].metric("Provider", "Ready" if status["provider_configured"] else "Missing")
                cols[1].metric("Tenant", "On" if status["enabled_for_tenant"] else "Off")
                cols[2].metric("Model", status.get("model") or "—")
                cols[3].metric("Fallback", "Rules" if status.get("fallback")=="deterministic" else (status.get("fallback") or "—").title())
                st.caption("Provider keys are configured only on the server. Without tenant authorization and server credentials, deterministic explanations remain active.")
                with st.form("llm_settings"):
                    enabled=st.checkbox("Authorize the configured provider for this organization",value=status["enabled_for_tenant"])
                    limit=st.number_input("Daily call limit",min_value=1,max_value=1000,value=int(status["daily_call_limit"]))
                    save=st.form_submit_button("Save LLM authorization")
                if save:
                    result=put("/api/settings/llm",{"enabled":enabled,"daily_call_limit":limit})
                    if result: st.success("LLM authorization updated."); st.rerun()
    with tabs[4]:
        if not roles.intersection({"OWNER","ADMIN"}): st.info("User management is limited to owners and administrators.")
        else:
            users=get("/api/tenant/users")
            if users:
                members=[{"Email":row.get("email"),"Name":row.get("full_name"),
                    "Roles":", ".join(row.get("roles",[])),"Active":row.get("is_active")} for row in users.get("items",[])]
                show_frame(members,empty="No other members.")
            with st.form("invite_user"):
                st.markdown("#### Add organization member")
                email=st.text_input("Email"); name=st.text_input("Full name")
                password=st.text_input("Temporary password (10+ characters)",type="password")
                roles_to_assign=st.multiselect("Roles",["ANALYST","APPROVER","ADMIN"],default=["ANALYST"])
                submit=st.form_submit_button("Add member")
            if submit:
                if not email.strip() or not name.strip() or len(password)<10 or not roles_to_assign:
                    st.error("Provide email, full name, a password of at least 10 characters, and one role.")
                else:
                    result=post("/api/tenant/users",{"email":email,"full_name":name,"password":password,"roles":roles_to_assign})
                    if result: st.success(f"Added {result['email']}."); st.rerun()


def reports_page():
    page_header("Reports", "Download tenant-scoped records and model evaluation summaries.")
    overview=get("/api/reports/overview")
    if overview:
        cols=st.columns(4)
        for col,key in zip(cols,["customers","analysis_runs","recommendations","simulated_executions"]): col.metric(key.replace("_"," ").title(),f"{overview[key]:,}")
        st.caption(overview["retention_claim"])
        st.markdown("#### Manually recorded outcomes")
        show_frame([{"Outcome status":k,"Count":v} for k,v in overview["recorded_outcomes"].items()],empty="No manual outcomes have been recorded.")
    models=get("/api/reports/models")
    if models:
        table=pd.DataFrame(models)
        st.download_button("Download model comparison CSV",table.to_csv(index=False),"aura_model_comparison.csv","text/csv")
    recommendations=get("/api/recommendations",limit=200)
    if recommendations:
        table=pd.DataFrame(recommendations["items"])
        st.download_button("Download tenant recommendations CSV",table.to_csv(index=False),"aura_recommendations.csv","text/csv")
    pdf=call("GET","/api/reports/tenant.pdf",raw=True,quiet=True)
    if pdf: st.download_button("Download tenant activity PDF",pdf,"aura_tenant_report.pdf","application/pdf")
    else: st.caption("PDF report is currently unavailable from the API.")


def main():
    if not st.session_state.get("access_token"):
        login_view(); return
    context=get("/api/auth/me")
    if not context: return
    st.session_state.user=context["user"]; st.session_state.tenant=context["tenant"]
    roles=set(context.get("roles",[]))
    st.sidebar.markdown('<div class="aura-brand">◈ AURA</div><div class="aura-brand-sub">AI-Powered User Retention Analytics</div>',unsafe_allow_html=True)
    st.sidebar.divider()
    st.sidebar.caption(context["tenant"]["name"])
    st.sidebar.caption(f"{context['user']['full_name']} · {' / '.join(sorted(roles))}")
    pages=["Overview","Customers","Customer profile","Model report","Causal explorer","Retention center","History & activity","Datasets","Reports","Settings"]
    page=st.sidebar.radio("Workspace",pages,key="active_page",label_visibility="collapsed")
    st.sidebar.divider()
    if st.sidebar.button("Refresh current page",width="stretch"): st.rerun()
    if st.sidebar.button("Sign out",width="stretch"):
        call("POST","/api/auth/logout",quiet=True)
        for key in ("access_token","user","tenant"): st.session_state.pop(key,None)
        st.rerun()
    st.sidebar.markdown("---")
    st.sidebar.caption("AURA v0.1 · Synthetic demonstration workspace")
    if page=="Overview": overview_page(context,roles)
    elif page=="Customers": customers_page()
    elif page=="Customer profile": customer_profile_page(roles)
    elif page=="Model report": model_page()
    elif page=="Causal explorer": causal_page()
    elif page=="Retention center": actions_page(roles)
    elif page=="History & activity": history_page()
    elif page=="Datasets": datasets_page(roles)
    elif page=="Reports": reports_page()
    elif page=="Settings": settings_page(context,roles)


main()
