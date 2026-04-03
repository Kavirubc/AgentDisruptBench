import streamlit as st
import json
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="AgentDisruptBench", page_icon="🛡️", layout="wide")

# ─── Minimal Light CSS ───
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    #MainMenu, footer, header { visibility: hidden; }

    .top-bar {
        padding: 1.2rem 0 0.8rem 0;
        border-bottom: 1px solid #e8e8e8;
        margin-bottom: 1.4rem;
    }
    .top-bar h1 { font-size: 1.5rem; font-weight: 700; margin: 0; color: #111; }
    .top-bar .sub { font-size: 0.82rem; color: #888; margin-top: 2px; }

    .kpi-row { display: flex; gap: 12px; margin-bottom: 1.4rem; }
    .kpi {
        flex: 1;
        background: #fff;
        border: 1px solid #eaeaea;
        border-radius: 8px;
        padding: 14px 16px;
        text-align: center;
    }
    .kpi .num { font-size: 1.6rem; font-weight: 700; line-height: 1.2; }
    .kpi .lbl { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.04em; color: #999; margin-top: 4px; }
    .green { color: #16a34a; }
    .amber { color: #d97706; }
    .red   { color: #dc2626; }
    .muted { color: #555; }

    .section-title {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #aaa;
        margin: 1.2rem 0 0.6rem 0;
    }

    .task-row {
        display: flex;
        align-items: center;
        padding: 10px 14px;
        border-bottom: 1px solid #f0f0f0;
        gap: 12px;
        font-size: 0.88rem;
    }
    .task-row:hover { background: #fafafa; }
    .task-name { flex: 1; font-weight: 500; color: #222; }
    .task-tag {
        font-size: 0.68rem;
        padding: 2px 8px;
        border-radius: 10px;
        font-weight: 500;
    }
    .tag-adversarial { background: #fef3c7; color: #92400e; }
    .tag-impossible  { background: #fce7f3; color: #9d174d; }
    .tag-handover    { background: #dbeafe; color: #1e40af; }
    .tag-standard    { background: #ecfdf5; color: #065f46; }

    .task-stat { min-width: 60px; text-align: right; color: #666; font-size: 0.82rem; }
    .task-score { min-width: 50px; text-align: right; font-weight: 700; font-size: 0.95rem; }

    .dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 6px; }
    .dot-pass { background: #16a34a; }
    .dot-fail { background: #dc2626; }

    .output-box {
        background: #f7f7f8;
        border: 1px solid #e5e5e5;
        border-radius: 8px;
        padding: 14px 16px;
        font-size: 0.85rem;
        line-height: 1.6;
        color: #333;
        white-space: pre-wrap;
        max-height: 280px;
        overflow-y: auto;
    }

    .trace-item {
        display: flex;
        align-items: center;
        padding: 7px 12px;
        font-size: 0.82rem;
        border-left: 3px solid #e5e5e5;
        margin-bottom: 4px;
        gap: 10px;
        background: #fafafa;
        border-radius: 0 6px 6px 0;
    }
    .trace-ok   { border-left-color: #16a34a; }
    .trace-warn { border-left-color: #f59e0b; }
    .trace-err  { border-left-color: #dc2626; }
    .trace-tool { font-weight: 600; min-width: 150px; }
    .trace-status { min-width: 80px; }
    .trace-lat { color: #999; margin-left: auto; font-size: 0.78rem; }

    .compare-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
    .compare-table th { text-align: left; padding: 8px 12px; border-bottom: 2px solid #eee; color: #888; font-weight: 600; font-size: 0.75rem; text-transform: uppercase; }
    .compare-table td { padding: 8px 12px; border-bottom: 1px solid #f0f0f0; }

    [data-testid="stSidebar"] { background: #fafafa; }
</style>
""", unsafe_allow_html=True)

RUNS_DIR = Path("runs")

def score_cls(s):
    if s >= 0.7: return "green"
    if s >= 0.4: return "amber"
    return "red"

def task_type(tid):
    if tid.startswith("adversarial"): return "adversarial"
    if tid.startswith("impossible"): return "impossible"
    if tid.startswith("handover"): return "handover"
    return "standard"

def norm_output(o):
    if isinstance(o, list):
        return "\n".join(item.get("text", str(item)) if isinstance(item, dict) else str(item) for item in o)
    return str(o) if o else ""

def load_run(p):
    tasks = []
    tl = p / "task_logs"
    if tl.exists():
        for f in sorted(tl.glob("*.json")):
            try:
                tasks.append(json.loads(f.read_text()))
            except Exception:
                pass
    summary = {}
    sp = p / "summary.json"
    if sp.exists():
        try: summary = json.loads(sp.read_text())
        except Exception: pass
    report = ""
    rp = p / "report.md"
    if rp.exists(): report = rp.read_text()
    return {"tasks": tasks, "summary": summary, "report": report}

# ─── Load ───
if not RUNS_DIR.exists() or not any(RUNS_DIR.iterdir()):
    st.warning("No runs found. Execute `adb evaluate` first.")
    st.stop()

runs = sorted([d for d in RUNS_DIR.iterdir() if d.is_dir()], key=lambda p: p.name, reverse=True)
cache = {r.name: load_run(r) for r in runs}

# ─── Sidebar ───
st.sidebar.markdown("**AgentDisruptBench**")
st.sidebar.caption("Resilience Evaluation Dashboard")
st.sidebar.markdown("---")
mode = st.sidebar.radio("Mode", ["Single Run", "Compare Runs"])

if mode == "Single Run":
    sel = st.sidebar.selectbox("Run", [r.name for r in runs])
    d = cache[sel]
    tasks = d["tasks"]

    if tasks:
        profile = tasks[0].get("profile_name", "—")
        agent = tasks[0].get("agent_id", "—")
        st.sidebar.markdown("---")
        st.sidebar.caption(f"Agent: `{agent}`")
        st.sidebar.caption(f"Profile: `{profile}`")
        st.sidebar.caption(f"Tasks: `{len(tasks)}`")

    # ─── Header ───
    st.markdown(f"""<div class="top-bar">
        <h1>🛡️ AgentDisruptBench</h1>
        <div class="sub">{sel} · Profile: {tasks[0].get('profile_name','—') if tasks else '—'}</div>
    </div>""", unsafe_allow_html=True)

    if not tasks:
        st.info("This run has no task logs.")
        st.stop()

    # ─── KPIs ───
    n = len(tasks)
    wins = sum(1 for t in tasks if t.get("success"))
    avg_s = sum(t.get("partial_score", 0) for t in tasks) / n
    tools = sum(t.get("total_tool_calls", 0) for t in tasks)
    disrupt = sum(t.get("disruptions_encountered", 0) for t in tasks)
    avg_dur = sum(t.get("duration_seconds", 0) for t in tasks) / n
    sr = wins / n

    st.markdown(f"""<div class="kpi-row">
        <div class="kpi"><div class="num muted">{n}</div><div class="lbl">Tasks</div></div>
        <div class="kpi"><div class="num {score_cls(sr)}">{sr:.0%}</div><div class="lbl">Success Rate</div></div>
        <div class="kpi"><div class="num {score_cls(avg_s)}">{avg_s:.2f}</div><div class="lbl">Avg Score</div></div>
        <div class="kpi"><div class="num muted">{tools}</div><div class="lbl">Tool Calls</div></div>
        <div class="kpi"><div class="num {'amber' if disrupt else 'green'}">{disrupt}</div><div class="lbl">Disruptions</div></div>
        <div class="kpi"><div class="num muted">{avg_dur:.1f}s</div><div class="lbl">Avg Duration</div></div>
    </div>""", unsafe_allow_html=True)

    # ─── Score Chart ───
    st.markdown('<div class="section-title">Score Distribution</div>', unsafe_allow_html=True)
    chart_df = pd.DataFrame([
        {"Task": t.get("task_id", ""), "Score": t.get("partial_score", 0), "Type": task_type(t.get("task_id", ""))}
        for t in tasks
    ])
    st.bar_chart(chart_df, x="Task", y="Score", color="Type", height=220)

    # ─── Task List ───
    st.markdown('<div class="section-title">Tasks</div>', unsafe_allow_html=True)

    fc1, fc2 = st.columns([1, 4])
    with fc1:
        filt = st.selectbox("Type", ["all", "adversarial", "impossible", "handover", "standard"], label_visibility="collapsed")

    ftasks = tasks if filt == "all" else [t for t in tasks if task_type(t.get("task_id", "")) == filt]

    # Table header
    st.markdown("""<div class="task-row" style="border-bottom:2px solid #eee;font-size:0.72rem;color:#aaa;text-transform:uppercase;letter-spacing:0.04em">
        <span style="width:16px"></span>
        <span class="task-name">Task</span>
        <span style="min-width:80px">Type</span>
        <span class="task-stat">Tools</span>
        <span class="task-stat">Disruptions</span>
        <span class="task-stat">Duration</span>
        <span class="task-score">Score</span>
    </div>""", unsafe_allow_html=True)

    for t in ftasks:
        tid = t.get("task_id", "?")
        sc = t.get("partial_score", 0)
        ok = t.get("success", False)
        tt = task_type(tid)
        dot = "dot-pass" if ok else "dot-fail"

        st.markdown(f"""<div class="task-row">
            <span class="dot {dot}"></span>
            <span class="task-name">{tid}</span>
            <span><span class="task-tag tag-{tt}">{tt}</span></span>
            <span class="task-stat">{t.get('total_tool_calls',0)}</span>
            <span class="task-stat">{t.get('disruptions_encountered',0)}</span>
            <span class="task-stat">{t.get('duration_seconds',0):.1f}s</span>
            <span class="task-score {score_cls(sc)}">{sc:.0%}</span>
        </div>""", unsafe_allow_html=True)

    # ─── Inspector ───
    st.markdown('<div class="section-title">Inspector</div>', unsafe_allow_html=True)
    pick = st.selectbox("Task", [t.get("task_id", "") for t in tasks], label_visibility="collapsed")
    sel_t = next((t for t in tasks if t.get("task_id") == pick), None)

    if sel_t:
        c1, c2 = st.columns([3, 1])
        with c1:
            st.caption("Agent Output")
            out = norm_output(sel_t.get("agent_output", ""))
            st.markdown(f'<div class="output-box">{out if out else "(empty)"}</div>', unsafe_allow_html=True)
        with c2:
            st.caption("Metrics")
            st.metric("Score", f"{sel_t.get('partial_score',0):.2f}")
            st.metric("Recovery", f"{sel_t.get('recovery_rate',0):.0%}")
            st.metric("Duration", f"{sel_t.get('duration_seconds',0):.1f}s")
            st.metric("Tools", sel_t.get("total_tool_calls", 0))

        # Traces
        traces = sel_t.get("traces", [])
        if traces:
            st.caption("Tool Timeline")
            for i, tr in enumerate(traces):
                tn = tr.get("tool_name", "?")
                ok = tr.get("observed_success", tr.get("real_success", True))
                dis = tr.get("disruption_fired", False)
                lat = tr.get("observed_latency_ms", tr.get("real_latency_ms", 0))

                cls = "trace-warn" if dis else ("trace-ok" if ok else "trace-err")
                status = "⚡ disrupted" if dis else ("✓ ok" if ok else "✗ fail")

                if isinstance(lat, float):
                    lat_str = f"{lat:.1f}ms"
                else:
                    lat_str = f"{lat}ms"

                st.markdown(f"""<div class="trace-item {cls}">
                    <span style="color:#bbb">{i+1}</span>
                    <span class="trace-tool">{tn}</span>
                    <span class="trace-status">{status}</span>
                    <span class="trace-lat">{lat_str}</span>
                </div>""", unsafe_allow_html=True)

            with st.expander("Raw JSON"):
                st.json(traces)

        # Error
        err = sel_t.get("error_msg")
        if err:
            st.error(f"Agent error: {err}")

    # Report
    if d["report"]:
        with st.expander("Markdown Report"):
            st.markdown(d["report"])

else:
    # ─── Compare Mode ───
    st.markdown("""<div class="top-bar">
        <h1>🛡️ AgentDisruptBench — Compare</h1>
        <div class="sub">Side-by-side run comparison</div>
    </div>""", unsafe_allow_html=True)

    picks = st.sidebar.multiselect("Runs", [r.name for r in runs], default=[r.name for r in runs[:min(3, len(runs))]])

    if len(picks) < 2:
        st.info("Pick at least 2 runs to compare.")
        st.stop()

    # Summary table
    rows = []
    for rn in picks:
        ts = cache[rn]["tasks"]
        if not ts: continue
        n = len(ts)
        w = sum(1 for t in ts if t.get("success"))
        rows.append({
            "Run": rn,
            "Profile": ts[0].get("profile_name", "?"),
            "Tasks": n,
            "Success": f"{w/n:.0%}",
            "Avg Score": round(sum(t.get("partial_score",0) for t in ts)/n, 3),
            "Tools": sum(t.get("total_tool_calls",0) for t in ts),
            "Disruptions": sum(t.get("disruptions_encountered",0) for t in ts),
            "Avg Dur": f"{sum(t.get('duration_seconds',0) for t in ts)/n:.1f}s",
        })

    if rows:
        st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Score chart
    st.markdown('<div class="section-title">Scores</div>', unsafe_allow_html=True)
    sdata = []
    for rn in picks:
        for t in cache[rn]["tasks"]:
            sdata.append({"Run": rn[-8:], "Task": t.get("task_id",""), "Score": t.get("partial_score",0)})
    if sdata:
        st.bar_chart(pd.DataFrame(sdata), x="Task", y="Score", color="Run", height=250)

    # Per-task compare
    st.markdown('<div class="section-title">Task Detail</div>', unsafe_allow_html=True)
    all_ids = sorted({t.get("task_id","") for rn in picks for t in cache[rn]["tasks"]})
    if all_ids:
        ct = st.selectbox("Task", all_ids)
        cols = st.columns(len(picks))
        for i, rn in enumerate(picks):
            m = next((t for t in cache[rn]["tasks"] if t.get("task_id") == ct), None)
            with cols[i]:
                st.caption(rn[-12:])
                if m:
                    st.metric("Score", f"{m.get('partial_score',0):.2f}")
                    st.metric("Tools", m.get("total_tool_calls",0))
                    st.metric("Time", f"{m.get('duration_seconds',0):.1f}s")
                    st.text_area("Output", norm_output(m.get("agent_output","")), height=120, key=f"c_{rn}_{ct}")
                else:
                    st.warning("Not in this run")
