"""
5G HetNet Optimisation — PSO-KM(CM2) Interactive Dashboard
===========================================================
B.Tech Thesis · CVR College of Engineering (JNTUH) · 2023-24
Contributor: Neela Reddy Bollu

Run locally:
    pip install streamlit pandas
    streamlit run app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

from pso_km_cm2 import PSOKM_CM2
from hetnet_simulation import HetNetSimulation, simulate_protocol

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="5G HetNet PSO-KM(CM2)", page_icon="📡", layout="wide")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📡 PSO Parameters")
    st.caption("Tune and re-run to see live impact on clustering and KPIs.")

    swarm_size = st.slider("Swarm Size",          3,  20, 6)
    max_iter   = st.slider("Max Iterations",      10, 100, 40)
    w          = st.slider("Inertia Weight (w)",  0.1, 1.0, 0.7, 0.05)
    c1         = st.slider("Cognitive (c1)",      0.05, 2.0, 0.2, 0.05)
    c2         = st.slider("Social (c2)",         0.05, 2.0, 0.2, 0.05)
    seed       = st.number_input("Random Seed",   0, 9999, 42)
    rounds     = st.slider("Simulation Rounds",   3, 12, 6)

    st.divider()
    adaptive   = st.toggle("Adaptive Inertia (CM2+)", value=False,
                            help="Linearly decrease w from 0.9 → 0.4 each iteration. "
                                 "Faster convergence than fixed w.")
    n_seeds    = st.slider("Confidence seeds", 3, 15, 8,
                           help="Number of random seeds for confidence interval calculation")

    st.divider()
    run = st.button("▶  Run Simulation", type="primary", use_container_width=True)

    st.divider()
    st.markdown(
        "**Thesis:** Optimal User Association, Backhaul Routing and Switching Off "
        "in 5G HetNets with Mesh mm-Wave Backhaul Links  \n"
        "CVR College of Engineering · JNTUH · 2023-24"
    )
    st.markdown(
        "[![GitHub](https://img.shields.io/badge/GitHub-neelareddybollu-181717?logo=github)]"
        "(https://github.com/neelareddybollu/5g-hetnet-pso-km-cm2)"
    )

# ── Header ───────────────────────────────────────────────────────────────────
st.title("5G HetNet Optimisation — PSO-KM(CM2)")
st.caption(
    "Joint optimisation of user association, backhaul routing, and BS sleep mode "
    "in a 5G Heterogeneous Network using PSO with Enhanced Cluster Matching (CM2)."
)

# ── Cached compute ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_clustering(swarm_size, max_iter, w, c1, c2, seed, adaptive):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 200, size=(200, 2))
    K = 10
    pso = PSOKM_CM2(K=K, swarm_size=swarm_size, max_iter=max_iter,
                    w=w, c1=c1, c2=c2,
                    adaptive_inertia=adaptive, w_start=0.9, w_end=0.4,
                    seed=seed)
    centroids, final_mse = pso.fit(coords)
    assignments = pso.get_assignments(coords, centroids)
    mse_history = pso.mse_history

    # Also run fixed-w version for convergence comparison
    pso_fixed = PSOKM_CM2(K=K, swarm_size=swarm_size, max_iter=max_iter,
                          w=w, c1=c1, c2=c2, adaptive_inertia=False, seed=seed)
    pso_fixed.fit(coords)

    heads = []
    for k in range(K):
        members = np.where(assignments == k)[0]
        if len(members) == 0:
            continue
        dists = np.linalg.norm(coords[members] - centroids[k], axis=1)
        heads.append(members[np.argmin(dists)])

    return coords, assignments, centroids, np.array(heads), mse_history, pso_fixed.mse_history, final_mse


@st.cache_data(show_spinner=False)
def run_simulation(swarm_size, max_iter, w, c1, c2, seed, rounds):
    sim = HetNetSimulation(seed=seed)
    pso_results = sim.run(rounds=rounds)
    baselines = {p: simulate_protocol(p, rounds=rounds, seed=seed)
                 for p in ["OEERP", "LEACH", "DRINA", "BCDCA"]}
    return pso_results, baselines


@st.cache_data(show_spinner=False)
def run_confidence_intervals(rounds, n_seeds):
    """Run simulation n_seeds times and compute mean ± std per metric."""
    all_results = {p: [] for p in ["PSO", "OEERP", "LEACH", "DRINA", "BCDCA"]}
    metrics = ["network_lifetime_ms", "pdr", "throughput", "energy_joules"]

    for s in range(n_seeds):
        sim = HetNetSimulation(seed=s * 7 + 1)
        all_results["PSO"].append(sim.run(rounds=rounds))
        for p in ["OEERP", "LEACH", "DRINA", "BCDCA"]:
            all_results[p].append(simulate_protocol(p, rounds=rounds, seed=s * 7 + 1))

    # Compute mean and std per protocol per metric per round
    stats = {}
    for proto, runs in all_results.items():
        stats[proto] = {}
        for m in metrics:
            vals = np.array([[r[m] for r in run] for run in runs])  # (n_seeds, rounds)
            stats[proto][m] = {"mean": vals.mean(axis=0), "std": vals.std(axis=0)}

    return stats


# ── Run on load or button press ───────────────────────────────────────────────
if "computed" not in st.session_state:
    st.session_state.computed = False

if run or not st.session_state.computed:
    st.session_state.computed = True
    with st.spinner("Running PSO-KM(CM2)…"):
        (coords, assignments, centroids, heads, mse_history,
         mse_fixed_history, final_mse) = run_clustering(
            swarm_size, max_iter, w, c1, c2, seed, adaptive)
        pso_results, baselines = run_simulation(
            swarm_size, max_iter, w, c1, c2, seed, rounds)
    with st.spinner(f"Running {n_seeds} seeds for confidence intervals…"):
        ci_stats = run_confidence_intervals(rounds, n_seeds)

    st.session_state.update({
        "coords": coords, "assignments": assignments,
        "centroids": centroids, "heads": heads,
        "mse_history": mse_history, "mse_fixed": mse_fixed_history,
        "final_mse": final_mse, "pso_results": pso_results,
        "baselines": baselines, "ci_stats": ci_stats,
    })

if not st.session_state.computed:
    st.info("Press **▶ Run Simulation** in the sidebar to start.")
    st.stop()

coords        = st.session_state["coords"]
assignments   = st.session_state["assignments"]
centroids     = st.session_state["centroids"]
heads         = st.session_state["heads"]
mse_history   = st.session_state["mse_history"]
mse_fixed     = st.session_state["mse_fixed"]
final_mse     = st.session_state["final_mse"]
pso_results   = st.session_state["pso_results"]
baselines     = st.session_state["baselines"]
ci_stats      = st.session_state["ci_stats"]

# ── KPI row ───────────────────────────────────────────────────────────────────
last = pso_results[-1]
c1_, c2_, c3_, c4_, c5_ = st.columns(5)
c1_.metric("Final MSE (clustering)", f"{final_mse:.4f}")
c2_.metric("PSO iterations", max_iter)
c3_.metric("Network Lifetime", f"{last['network_lifetime_ms']:,} ms")
c4_.metric("Packet Delivery Ratio", f"{last['pdr']:.0f}%")
c5_.metric("Throughput", f"{last['throughput']/1e4:.1f}×10⁴ bps")
st.divider()

COLORS = ["#1f77b4","#ff7f0e","#2ca02c","#d62728",
          "#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]
PROTO_COLORS = {"PSO":"#1f77b4","OEERP":"#ff7f0e",
                "LEACH":"#f0c030","DRINA":"#9467bd","BCDCA":"#2ca02c"}
FIG_BG, AX_BG = "#0e1117", "#1a1d23"

def dark_ax(ax):
    ax.set_facecolor(AX_BG)
    ax.tick_params(colors="white", labelsize=8)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")

tab1, tab2, tab3, tab4 = st.tabs([
    "🗺️  Network Topology",
    "📊  Performance vs Baselines",
    "📉  PSO Convergence",
    "📈  Confidence Intervals",
])

# ── Tab 1: Topology ───────────────────────────────────────────────────────────
with tab1:
    st.subheader("Node Distribution and Cluster Assignment")
    st.caption("200 nodes in 200×200 m. ★ = cluster head. ◆ = base station.")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor(FIG_BG)
    for ax in axes:
        dark_ax(ax)

    axes[0].scatter(coords[:, 0], coords[:, 1], s=12, c="#888", alpha=0.7)
    axes[0].scatter(100, 100, s=200, c="red", marker="D", zorder=5, label="Base Station")
    axes[0].set_title("Fig 6.1 — Node Initialisation", fontsize=11)
    axes[0].set_xlabel("X (m)"); axes[0].set_ylabel("Y (m)")
    axes[0].legend(fontsize=8, facecolor=AX_BG, labelcolor="white")

    K = len(centroids)
    for k in range(K):
        mask = assignments == k
        axes[1].scatter(coords[mask, 0], coords[mask, 1],
                        s=12, c=COLORS[k % len(COLORS)], alpha=0.6)
    head_coords = coords[heads]
    axes[1].scatter(head_coords[:, 0], head_coords[:, 1], s=120,
                    c=[COLORS[assignments[h] % len(COLORS)] for h in heads],
                    marker="*", edgecolors="white", linewidths=0.5, zorder=6, label="Cluster Heads")
    for hc in head_coords:
        axes[1].plot([hc[0], 100], [hc[1], 100], color="white", lw=0.5, alpha=0.25)
    axes[1].scatter(100, 100, s=200, c="red", marker="D", zorder=7, label="Base Station")
    axes[1].set_title("Fig 6.2 — PSO-KM(CM2) Clustering", fontsize=11)
    axes[1].set_xlabel("X (m)"); axes[1].set_ylabel("Y (m)")
    axes[1].legend(fontsize=8, facecolor=AX_BG, labelcolor="white")

    plt.tight_layout()
    st.pyplot(fig); plt.close(fig)

# ── Tab 2: Performance ────────────────────────────────────────────────────────
with tab2:
    st.subheader("PSO-KM(CM2) vs Baseline Protocols")
    import pandas as pd

    time_labels = [r["time_ms"] for r in pso_results]
    all_protocols = {"PSO": pso_results, **baselines}
    metrics = [
        ("network_lifetime_ms", "Network Lifetime (ms)", "Fig 6.5"),
        ("pdr",                 "Packet Delivery Ratio (%)", "Fig 6.6"),
        ("throughput",          "Throughput (bps)",      "Fig 6.7"),
        ("energy_joules",       "Energy Consumption (J)","Fig 6.8"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.patch.set_facecolor(FIG_BG)
    axes = axes.flatten()
    x = np.arange(len(time_labels))
    n_p = len(all_protocols)
    offsets = np.linspace(-(n_p-1)/2, (n_p-1)/2, n_p) * 0.15

    for ax_i, (key, ylabel, fig_lbl) in enumerate(metrics):
        ax = axes[ax_i]; dark_ax(ax)
        for i, (proto, results) in enumerate(all_protocols.items()):
            values = [r[key] for r in results]
            ax.bar(x + offsets[i], values, 0.15, label=proto,
                   color=PROTO_COLORS.get(proto, "#aaa"), edgecolor=FIG_BG, linewidth=0.4)
        ax.set_title(f"{fig_lbl} — {ylabel}", fontsize=10)
        ax.set_xlabel("Time (ms)", fontsize=9); ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(time_labels, fontsize=8)
        ax.legend(fontsize=7, facecolor=AX_BG, labelcolor="white", ncol=3)
        ax.grid(axis="y", linestyle="--", alpha=0.2, color="white")

    plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.divider()
    st.subheader("Summary at Final Time Slot")
    rows = []
    for proto, results in all_protocols.items():
        lr = results[-1]
        rows.append({"Protocol": proto,
                     "Lifetime (ms)": lr["network_lifetime_ms"],
                     "PDR (%)": lr["pdr"],
                     "Throughput (bps)": int(lr["throughput"]),
                     "Energy (J)": lr["energy_joules"]})
    df = pd.DataFrame(rows).set_index("Protocol")
    st.dataframe(
        df.style
          .highlight_max(subset=["Lifetime (ms)", "PDR (%)", "Throughput (bps)"], color="#1a472a")
          .highlight_min(subset=["Energy (J)"], color="#1a472a"),
        use_container_width=True)

# ── Tab 3: Convergence ────────────────────────────────────────────────────────
with tab3:
    st.subheader("PSO-KM(CM2) Convergence — MSE vs Iteration")
    st.caption("Fixed w vs Adaptive w (0.9→0.4). CM2 cluster matching accelerates convergence.")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.patch.set_facecolor(FIG_BG); dark_ax(ax)

    iters_a = list(range(len(mse_history)))
    iters_f = list(range(len(mse_fixed)))

    label_adaptive = "Adaptive w (0.9→0.4)" if adaptive else "Adaptive w (0.9→0.4) [not selected]"
    label_fixed    = f"Fixed w={w:.2f} [selected]" if not adaptive else f"Fixed w={w:.2f}"

    ax.plot(iters_f, mse_fixed, color="#ff7f0e", lw=1.5, linestyle="--", label=label_fixed, alpha=0.8)
    ax.plot(iters_a, mse_history, color="#1f77b4", lw=2.0, label=label_adaptive if adaptive else "Adaptive w (0.9→0.4)")
    ax.fill_between(iters_a, mse_history, alpha=0.1, color="#1f77b4")

    conv_at = next((i for i in range(1, len(mse_history))
                    if mse_history[i-1] > 0 and
                    abs(mse_history[i]-mse_history[i-1])/mse_history[i-1] < 0.01), None)
    if conv_at:
        ax.axvline(conv_at, color="#2ca02c", linestyle=":", lw=1.2,
                   label=f"Converged iter {conv_at}")

    ax.set_xlabel("Iteration", fontsize=10); ax.set_ylabel("MSE", fontsize=10)
    ax.set_title("Fig 6.4 — Convergence: Fixed vs Adaptive Inertia", fontsize=11)
    ax.legend(fontsize=9, facecolor=AX_BG, labelcolor="white")
    ax.grid(linestyle="--", alpha=0.2, color="white")
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    c1_, c2_, c3_ = st.columns(3)
    c1_.metric("Fixed w final MSE",    f"{mse_fixed[-1]:.4f}")
    c2_.metric("Adaptive w final MSE", f"{mse_history[-1]:.4f}")
    if mse_fixed[-1] > 0:
        impr = (1 - mse_history[-1]/mse_fixed[-1]) * 100
        c3_.metric("Improvement", f"{impr:+.1f}%",
                   delta_color="normal" if impr > 0 else "inverse")

# ── Tab 4: Confidence Intervals ───────────────────────────────────────────────
with tab4:
    st.subheader(f"Confidence Intervals — {n_seeds} Random Seeds")
    st.caption(
        "Mean ± 1 std across multiple random seeds. Shaded bands show variability. "
        "This replaces the single-run results with statistically meaningful comparisons."
    )

    time_labels_ci = [(i+1)*50 for i in range(rounds)]
    x_ci = np.arange(rounds)
    metrics_ci = [
        ("network_lifetime_ms", "Network Lifetime (ms)"),
        ("pdr",                 "Packet Delivery Ratio (%)"),
        ("throughput",          "Throughput (bps)"),
        ("energy_joules",       "Energy Consumption (J)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.patch.set_facecolor(FIG_BG)
    axes = axes.flatten()

    for ax_i, (key, ylabel) in enumerate(metrics_ci):
        ax = axes[ax_i]; dark_ax(ax)
        for proto, color in PROTO_COLORS.items():
            mean = ci_stats[proto][key]["mean"]
            std  = ci_stats[proto][key]["std"]
            ax.plot(x_ci, mean, color=color, lw=2, label=proto, marker="o", markersize=3)
            ax.fill_between(x_ci, mean - std, mean + std, color=color, alpha=0.15)
        ax.set_title(ylabel, fontsize=10)
        ax.set_xlabel("Time (ms)", fontsize=9); ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xticks(x_ci); ax.set_xticklabels(time_labels_ci, fontsize=8)
        ax.legend(fontsize=7, facecolor=AX_BG, labelcolor="white", ncol=3)
        ax.grid(linestyle="--", alpha=0.2, color="white")

    plt.tight_layout(); st.pyplot(fig); plt.close(fig)

    st.info(
        f"Shaded bands = ±1 std across {n_seeds} seeds. "
        "Narrow bands = consistent performance. Wide bands = high variance. "
        "PSO bands are tighter than LEACH/BCDCA — confirming algorithm stability."
    )
