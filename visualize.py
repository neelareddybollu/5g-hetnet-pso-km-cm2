"""
Visualisation — Reproduces thesis Figures 6.1 – 6.8
=====================================================
Generates all plots from Chapter 6 (Results) of:
"Optimal User Association, Backhaul Routing and Switching Off in
5G HetNets with Mesh mm-Wave Backhaul Links"

Figures produced:
  fig_6_1_node_init.png         — Scatter: random node initialisation
  fig_6_2_clustering.png        — Scatter: PSO-KM cluster assignment
  fig_6_3_node_selection.png    — Scatter: cluster heads + backhaul path
  fig_6_4_pso_performance.png   — Bar: PSO accuracy/sensitivity/specificity
  fig_6_5_network_lifetime.png  — Bar: network lifetime comparison
  fig_6_6_pdr.png               — Bar: packet delivery ratio comparison
  fig_6_7_throughput.png        — Bar: throughput comparison
  fig_6_8_energy.png            — Bar: energy consumption comparison
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from hetnet_simulation import HetNetSimulation, simulate_protocol

OUTPUT_DIR = 'figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)

PROTOCOLS  = ['PSO', 'OEERP', 'LEACH', 'DRINA', 'BCDCA']
TIME_SLOTS = [50, 100, 150, 200, 250, 300]
COLORS     = ['#1f77b4', '#ff7f0e', '#f0c030', '#9467bd', '#2ca02c']


# ---------------------------------------------------------------------------
# Helper: grouped bar chart
# ---------------------------------------------------------------------------

def grouped_bar(ax, data_dict: dict, ylabel: str, title: str, yscale=None):
    """
    data_dict: {protocol: [value_per_timeslot]}
    """
    x = np.arange(len(TIME_SLOTS))
    n = len(data_dict)
    width = 0.15
    offset = np.linspace(-(n - 1) / 2, (n - 1) / 2, n) * width

    for i, (protocol, values) in enumerate(data_dict.items()):
        ax.bar(x + offset[i], values, width, label=protocol,
               color=COLORS[i], edgecolor='white', linewidth=0.5)

    ax.set_xlabel('Time in Milli Seconds', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold', color='steelblue')
    ax.set_xticks(x)
    ax.set_xticklabels(TIME_SLOTS)
    ax.legend(fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    if yscale:
        ax.set_yscale(yscale)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ---------------------------------------------------------------------------
# Fig 6.1 — Node Initialisation
# ---------------------------------------------------------------------------

def plot_node_init(sim: HetNetSimulation):
    fig, ax = plt.subplots(figsize=(6, 6))
    xs = [n.x for n in sim.nodes]
    ys = [n.y for n in sim.nodes]
    ax.scatter(xs, ys, s=80, facecolors='none', edgecolors='royalblue', linewidths=1.2)
    ax.set_xlim(0, 200); ax.set_ylim(0, 200)
    ax.set_xlabel('X'); ax.set_ylabel('Y')
    ax.set_title('Fig. 6.1  Initialization of Nodes', fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig_6_1_node_init.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Fig 6.2 — Clustering
# ---------------------------------------------------------------------------

def plot_clustering(sim: HetNetSimulation):
    from pso_km_cm2 import PSOKM_CM2
    alive = [n for n in sim.nodes if n.alive]
    coords = np.array([[n.x, n.y] for n in alive])
    K = 10
    pso = PSOKM_CM2(K=K, swarm_size=6, max_iter=40, seed=42)
    centroids, _ = pso.fit(coords)
    assignments = pso.get_assignments(coords, centroids)

    cmap = plt.cm.get_cmap('tab20', K)
    fig, ax = plt.subplots(figsize=(6, 6))

    for k in range(K):
        mask = assignments == k
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   s=60, color=cmap(k), alpha=0.8)
        # Draw cluster boundary circle
        r = np.max(np.linalg.norm(coords[mask] - centroids[k], axis=1)) if mask.any() else 10
        circle = plt.Circle(centroids[k], r, fill=False,
                            edgecolor='black', linewidth=1.0, alpha=0.5)
        ax.add_patch(circle)

    ax.set_xlim(0, 200); ax.set_ylim(0, 200)
    ax.set_xlabel('X'); ax.set_ylabel('Y')
    ax.set_title('Fig. 6.2  Clustering (PSO-KM)', fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig_6_2_clustering.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Fig 6.3 — Node Selection (cluster heads + backhaul path)
# ---------------------------------------------------------------------------

def plot_node_selection(sim: HetNetSimulation):
    from pso_km_cm2 import PSOKM_CM2
    alive = [n for n in sim.nodes if n.alive]
    coords = np.array([[n.x, n.y] for n in alive])
    K = 10
    pso = PSOKM_CM2(K=K, swarm_size=6, max_iter=40, seed=42)
    centroids, _ = pso.fit(coords)
    assignments = pso.get_assignments(coords, centroids)

    # Identify cluster heads (closest to centroid)
    head_indices = []
    for k in range(K):
        members = np.where(assignments == k)[0]
        if len(members) == 0:
            continue
        dists = np.linalg.norm(coords[members] - centroids[k], axis=1)
        head_indices.append(members[np.argmin(dists)])

    fig, ax = plt.subplots(figsize=(6, 6))

    # Member nodes
    non_heads = [i for i in range(len(alive)) if i not in head_indices]
    ax.scatter(coords[non_heads, 0], coords[non_heads, 1],
               s=60, color='cyan', edgecolors='steelblue', linewidths=0.8,
               zorder=2, label='Member node')

    # Cluster circles
    cmap = plt.cm.get_cmap('tab20', K)
    for k in range(K):
        mask = assignments == k
        if not mask.any():
            continue
        r = np.max(np.linalg.norm(coords[mask] - centroids[k], axis=1))
        circle = plt.Circle(centroids[k], r, fill=False,
                            edgecolor='black', linewidth=1.0, alpha=0.4)
        ax.add_patch(circle)

    # Cluster heads
    head_coords = coords[head_indices]
    ax.scatter(head_coords[:, 0], head_coords[:, 1],
               s=120, color='magenta', edgecolors='darkred', linewidths=1.2,
               zorder=4, label='Cluster head')

    # Backhaul path: connect heads in a chain to BS (100,100)
    BS = np.array([100, 100])
    sorted_heads = sorted(head_indices,
                          key=lambda i: np.linalg.norm(coords[i] - BS))
    path = [BS] + [coords[i] for i in sorted_heads[:5]]
    for j in range(len(path) - 1):
        ax.plot([path[j][0], path[j + 1][0]],
                [path[j][1], path[j + 1][1]],
                'k-', linewidth=2, zorder=3)

    # Label cluster heads
    for rank, i in enumerate(sorted_heads[:8], 1):
        ax.annotate(str(rank), coords[i], fontsize=7, ha='center', va='center',
                    color='white', fontweight='bold')

    ax.set_xlim(0, 200); ax.set_ylim(0, 200)
    ax.set_xlabel('X'); ax.set_ylabel('Y')
    ax.set_title('Fig. 6.3  Node Selection & Backhaul Path', fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.3)
    fig.tight_layout()
    path_out = os.path.join(OUTPUT_DIR, 'fig_6_3_node_selection.png')
    fig.savefig(path_out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path_out}")


# ---------------------------------------------------------------------------
# Fig 6.4 — PSO Performance Metrics
# ---------------------------------------------------------------------------

def plot_pso_performance():
    fig, ax = plt.subplots(figsize=(5, 5))
    metrics = ['Accuracy', 'Sensitivity', 'Specificity']
    values  = [88, 92, 85]          # from thesis Fig 6.4
    colors  = ['red', 'green', 'blue']

    bars = ax.bar(metrics, values, color=colors, width=0.5, edgecolor='white')
    ax.set_ylim(0, 110)
    ax.set_ylabel('Percentage (%)', fontsize=11)
    ax.set_title('Fig. 6.4  Performance Measured for PSO', fontweight='bold')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{val}%', ha='center', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'fig_6_4_pso_performance.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Figs 6.5 – 6.8 — Protocol comparison charts
# ---------------------------------------------------------------------------

def build_comparison_data(pso_results: list[dict]) -> dict:
    """Pull per-timeslot values from PSO results and baselines."""
    baselines = {p: simulate_protocol(p, rounds=6) for p in ['OEERP', 'LEACH', 'DRINA', 'BCDCA']}

    def extract(key, scale=1.0):
        out = {'PSO': [r[key] * scale for r in pso_results]}
        for p, res in baselines.items():
            out[p] = [r[key] * scale for r in res]
        return out

    return {
        'network_lifetime_ms': extract('network_lifetime_ms'),
        'pdr':                 extract('pdr'),
        'throughput':          extract('throughput'),
        'energy_joules':       extract('energy_joules'),
    }


def plot_comparison_charts(pso_results: list[dict]):
    data = build_comparison_data(pso_results)

    specs = [
        ('fig_6_5_network_lifetime.png',
         data['network_lifetime_ms'],
         'Network Lifetime (ms)',
         'Fig. 6.5  Overall Network Lifetime at Different Time Slots'),

        ('fig_6_6_pdr.png',
         data['pdr'],
         'Packet Delivery Ratio (PDR %)',
         'Fig. 6.6  Packet Delivery Ratio at Different Time Slots'),

        ('fig_6_7_throughput.png',
         data['throughput'],
         'Throughput (bps)',
         'Fig. 6.7  Throughput at Different Time Slots'),

        ('fig_6_8_energy.png',
         data['energy_joules'],
         'Total Energy Consumption (Joules)',
         'Fig. 6.8  Total Energy Consumption at Different Time Slots'),
    ]

    for filename, chart_data, ylabel, title in specs:
        fig, ax = plt.subplots(figsize=(8, 5))
        grouped_bar(ax, chart_data, ylabel, title)
        fig.tight_layout()
        path = os.path.join(OUTPUT_DIR, filename)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")
