"""
5G HetNet Optimisation — PSO-KM(CM2) Interactive Dashboard
===========================================================
B.Tech Thesis · CVR College of Engineering (JNTUH) · 2023-24
Contributor: Neela Reddy Bollu

Run locally:
    pip install streamlit
    streamlit run app.py

Live demo: https://github.com/neelareddybollu/5g-hetnet-pso-km-cm2
"""

import sys
import os

# Ensure local modules are importable when run from any working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st

from pso_km_cm2 import PSOKM_CM2
from hetnet_simulation import HetNetSimulation, simulate_protocol

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="5G HetNet PSO-KM(CM2)",
    page_icon="📡",
    layout="wide",
)

# ──────────────────────────────────────────────
# Sidebar — PSO parameter controls
# ──────────────────────────────────────────────
with st.sidebar:
    st.title("📡 PSO Parameters")
    st.caption("Tune and re-run to see live impact on clustering and network KPIs.")

    swarm_size = st.slider("Swarm Size", min_value=3, max_value=20, value=6,
                           help="Number of particles (thesis default: 6)")
    max_iter   = st.slider("Max Iterations", min_value=10, max_value=100, value=40,
                           help="PSO iteration budget (thesis default: 40)")
    w          = st.slider("Inertia Weight (w)", min_value=0.1, max_value=1.0,
                           value=0.7, step=0.05,
                           help="Controls momentum of particle movement (thesis default: 0.7)")
    c1         = st.slider("Cognitive Factor (c1)", min_value=0.05, max_value=2.0,
                           value=0.2, step=0.05,
                           help="Pull towards personal best (thesis default: 0.2)")
    c2         = st.slider("Social Factor (c2)", min_value=0.05, max_value=2.0,
                           value=0.2, step=0.05,
                           help="Pull towards global best (thesis default: 0.2)")
    seed       = st.number_input("Random Seed", min_value=0, max_value=9999,
                                 value=42, step=1)
    rounds     = st.slider("Simulation Rounds", min_value=3, max_value=12,
                           value=6, help="Each round = 50 ms time slot")

    st.divider()
    run = st.button("▶  Run Simulation", type="primary", use_container_width=True)
    st.caption("Results are cached — change any slider and re-run.")

    st.divider()
    st.markdown(
        "**Thesis**  \nOptimal User Association, Backhaul Routing and "
        "Switching Off in 5G HetNets with Mesh mm-Wave Backhaul Links  \n"
        "CVR College of Engineering, JNTUH, 2023-24"
    )
    st.markdown(
        "[![GitHub](https://img.shields.io/badge/GitHub-neelareddybollu-181717?logo=github)]"
        "(https://github.com/neelareddybollu/5g-hetnet-pso-km-cm2)"
    )

# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.title("5G HetNet Optimisation — PSO-KM(CM2)")
st.caption(
    "Joint optimisation of user association, backhaul routing, and BS sleep mode "
    "in a 5G Heterogeneous Network using Particle Swarm Optimisation with "
    "Enhanced Cluster Matching (CM2)."
)

# ──────────────────────────────────────────────
# Cached computation functions
# ──────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_clustering(swarm_size, max_iter, w, c1, c2, seed):
    """Run PSO-KM(CM2) on 200 random nodes and return results."""
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 200, size=(200, 2))

    K = 10
    pso = PSOKM_CM2(K=K, swarm_size=swarm_size, max_iter=max_iter,
                    w=w, c1=c1, c2=c2, seed=seed)
    centroids, final_mse = pso.fit(coords)
    assignments = pso.get_assignments(coords, centroids)
    mse_history = pso.mse_history

    # Select cluster heads (node closest to each centroid)
    heads = []
    for k in range(K):
        members = np.where(assignments == k)[0]
        if len(members) == 0:
            continue
        dists = np.linalg.norm(coords[members] - centroids[k], axis=1)
        heads.append(members[np.argmin(dists)])

    return coords, assignments, centroids, np.array(heads), mse_history, final_mse


@st.cache_data(show_spinner=False)
def run_simulation(swarm_size, max_iter, w, c1, c2, seed, rounds):
    """Run full HetNet simulation for PSO and all baselines."""
    sim = HetNetSimulation(seed=seed)
    pso_results = sim.run(rounds=rounds)

    baselines = {}
    for proto in ["OEERP", "LEACH", "DRINA", "BCDCA"]:
        baselines[proto] = simulate_protocol(proto, rounds=rounds, seed=seed)

    return pso_results, baselines


# ──────────────────────────────────────────────
# Run on first load or when button pressed
# ──────────────────────────────────────────────
if "computed" not in st.session_state:
    st.session_state.computed = False

if run or not st.session_state.computed:
    st.session_state.computed = True
    with st.spinner("Running PSO-KM(CM2) simulation…"):
        (coords, assignments, centroids, heads,
         mse_history, final_mse) = run_clustering(
            swarm_size, max_iter, w, c1, c2, seed)
        pso_results, baselines = run_simulation(
            swarm_size, max_iter, w, c1, c2, seed, rounds)
    st.session_state.update({
        "coords": coords, "assignments": assignments,
        "centroids": centroids, "heads": heads,
        "mse_history": mse_history, "final_mse": final_mse,
        "pso_results": pso_results, "baselines": baselines,
    })

# Guard: show placeholder if not yet computed
if not st.session_state.computed:
    st.info("Press **▶ Run Simulation** in the sidebar to start.")
    st.stop()

# Unpack state
coords       = st.session_state["coords"]
assignments  = st.session_state["assignments"]
centroids    = st.session_state["centroids"]
heads        = st.session_state["heads"]
mse_history  = st.session_state["mse_history"]
final_mse    = st.session_state["final_mse"]
pso_results  = st.session_state["pso_results"]
baselines    = st.session_state["baselines"]

# ──────────────────────────────────────────────
# KPI summary row
# ──────────────────────────────────────────────
last = pso_results[-1]
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Final MSE (clustering)", f"{final_mse:.4f}")
col2.metric("PSO iterations", max_iter)
col3.metric("Network Lifetime", f"{last['network_lifetime_ms']:,} ms")
col4.metric("Packet Delivery Ratio", f"{last['pdr']:.0f}%")
col5.metric("Throughput", f"{last['throughput']/1e4:.1f}×10⁴ bps")

st.divider()

# ──────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🗺️  Network Topology",
    "📊  Performance vs Baselines",
    "📉  PSO Convergence",
])

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
          "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
          "#bcbd22", "#17becf"]
PROTO_COLORS = {
    "PSO":   "#1f77b4",
    "OEERP": "#ff7f0e",
    "LEACH": "#f0c030",
    "DRINA": "#9467bd",
    "BCDCA": "#2ca02c",
}

# ── Tab 1: Network Topology ──────────────────
with tab1:
    st.subheader("Node Distribution and Cluster Assignment")
    st.caption(
        "200 nodes distributed in a 200 × 200 m area. "
        "Colours = cluster assignment. ★ = cluster head. ◆ = base station (100, 100)."
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor("#0e1117")

    for ax in axes:
        ax.set_facecolor("#1a1d23")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

    # Left: raw node placement
    axes[0].scatter(coords[:, 0], coords[:, 1],
                    s=12, c="#888", alpha=0.7)
    axes[0].scatter(100, 100, s=200, c="red", marker="D", zorder=5, label="Base Station")
    axes[0].set_title("Fig 6.1 — Node Initialisation", fontsize=11)
    axes[0].set_xlabel("X (m)")
    axes[0].set_ylabel("Y (m)")
    axes[0].legend(fontsize=8, facecolor="#1a1d23", labelcolor="white")

    # Right: clustered
    K = len(centroids)
    for k in range(K):
        mask = assignments == k
        axes[1].scatter(coords[mask, 0], coords[mask, 1],
                        s=12, c=COLORS[k % len(COLORS)], alpha=0.6)

    # Cluster heads
    head_coords = coords[heads]
    axes[1].scatter(head_coords[:, 0], head_coords[:, 1],
                    s=120, c=[COLORS[assignments[h] % len(COLORS)] for h in heads],
                    marker="*", edgecolors="white", linewidths=0.5,
                    zorder=6, label="Cluster Heads")

    # Backhaul lines from heads to BS
    for hc in head_coords:
        axes[1].plot([hc[0], 100], [hc[1], 100],
                     color="white", lw=0.5, alpha=0.25, zorder=3)

    axes[1].scatter(100, 100, s=200, c="red", marker="D", zorder=7, label="Base Station")
    axes[1].set_title("Fig 6.2 — PSO-KM(CM2) Clustering", fontsize=11)
    axes[1].set_xlabel("X (m)")
    axes[1].set_ylabel("Y (m)")
    axes[1].legend(fontsize=8, facecolor="#1a1d23", labelcolor="white")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ── Tab 2: Performance vs Baselines ─────────
with tab2:
    st.subheader("PSO-KM(CM2) vs Baseline Protocols")
    st.caption("Each time slot = 50 ms. Lower energy = better; higher values = better for other metrics.")

    time_labels = [r["time_ms"] for r in pso_results]
    all_protocols = {"PSO": pso_results, **baselines}

    metrics = [
        ("network_lifetime_ms", "Network Lifetime (ms)",     "Fig 6.5"),
        ("pdr",                 "Packet Delivery Ratio (%)", "Fig 6.6"),
        ("throughput",          "Throughput (bps)",          "Fig 6.7"),
        ("energy_joules",       "Energy Consumption (J)",    "Fig 6.8"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.patch.set_facecolor("#0e1117")
    axes = axes.flatten()

    x = np.arange(len(time_labels))
    width = 0.15
    n_proto = len(all_protocols)
    offsets = np.linspace(-(n_proto - 1) / 2, (n_proto - 1) / 2, n_proto) * width

    for ax_idx, (key, ylabel, fig_label) in enumerate(metrics):
        ax = axes[ax_idx]
        ax.set_facecolor("#1a1d23")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")

        for i, (proto, results) in enumerate(all_protocols.items()):
            values = [r[key] for r in results]
            ax.bar(x + offsets[i], values, width,
                   label=proto,
                   color=PROTO_COLORS.get(proto, "#aaa"),
                   edgecolor="#0e1117", linewidth=0.4)

        ax.set_title(f"{fig_label} — {ylabel}", fontsize=10, color="white")
        ax.set_xlabel("Time (ms)", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(time_labels, fontsize=8)
        ax.legend(fontsize=7, facecolor="#1a1d23", labelcolor="white", ncol=3)
        ax.grid(axis="y", linestyle="--", alpha=0.25, color="white")
        ax.tick_params(colors="white", labelsize=8)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Summary table
    st.divider()
    st.subheader("Summary at Final Time Slot")
    import pandas as pd

    summary_rows = []
    for proto, results in all_protocols.items():
        last_r = results[-1]
        summary_rows.append({
            "Protocol":           proto,
            "Network Lifetime (ms)": last_r["network_lifetime_ms"],
            "PDR (%)":            last_r["pdr"],
            "Throughput (bps)":   int(last_r["throughput"]),
            "Energy (J)":         last_r["energy_joules"],
        })

    df = pd.DataFrame(summary_rows).set_index("Protocol")
    st.dataframe(
        df.style.highlight_max(
            subset=["Network Lifetime (ms)", "PDR (%)", "Throughput (bps)"],
            color="#1a472a"
        ).highlight_min(
            subset=["Energy (J)"],
            color="#1a472a"
        ),
        use_container_width=True,
    )

# ── Tab 3: PSO Convergence ───────────────────
with tab3:
    st.subheader("PSO-KM(CM2) Convergence — MSE vs Iteration")
    st.caption(
        "Mean Squared Error (MSE) of cluster assignment across PSO iterations. "
        "CM2 cluster matching accelerates convergence by resolving centroid-ordering ambiguity."
    )

    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#1a1d23")

    iters = list(range(len(mse_history)))
    ax.plot(iters, mse_history, color="#1f77b4", lw=2, label="PSO-KM(CM2)")
    ax.fill_between(iters, mse_history, alpha=0.15, color="#1f77b4")

    # Mark convergence point (< 1% change)
    converged_at = None
    for i in range(1, len(mse_history)):
        if mse_history[i - 1] > 0:
            change = abs(mse_history[i] - mse_history[i - 1]) / mse_history[i - 1]
            if change < 0.01 and converged_at is None:
                converged_at = i

    if converged_at:
        ax.axvline(converged_at, color="#ff7f0e", linestyle="--", lw=1.2,
                   label=f"Convergence (iter {converged_at})")
        ax.annotate(
            f"  <1% change\n  iter {converged_at}",
            xy=(converged_at, mse_history[converged_at]),
            xytext=(converged_at + max_iter * 0.05, mse_history[converged_at] * 1.1),
            color="#ff7f0e", fontsize=8,
        )

    ax.set_xlabel("Iteration", color="white", fontsize=10)
    ax.set_ylabel("MSE (clustering fitness)", color="white", fontsize=10)
    ax.set_title("Fig 6.4 — PSO Convergence (CM2 Enhanced)", color="white", fontsize=11)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.legend(fontsize=9, facecolor="#1a1d23", labelcolor="white")
    ax.grid(linestyle="--", alpha=0.2, color="white")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # Stats
    c1_, c2_, c3_ = st.columns(3)
    c1_.metric("Initial MSE",  f"{mse_history[0]:.2f}")
    c2_.metric("Final MSE",    f"{mse_history[-1]:.4f}")
    if mse_history[0] > 0:
        reduction = (1 - mse_history[-1] / mse_history[0]) * 100
        c3_.metric("MSE Reduction", f"{reduction:.1f}%")

    if converged_at:
        st.success(
            f"Converged at iteration **{converged_at}** (out of {max_iter}). "
            f"Try reducing Max Iterations to **{converged_at + 5}** — same result, faster runtime."
        )
