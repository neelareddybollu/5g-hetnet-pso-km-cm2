"""
main.py — Entry point
=====================
Run the full simulation and generate all thesis result figures.

Usage:
    python main.py
    python main.py --rounds 6 --seed 42

Outputs:
    figures/fig_6_1_node_init.png
    figures/fig_6_2_clustering.png
    figures/fig_6_3_node_selection.png
    figures/fig_6_4_pso_performance.png
    figures/fig_6_5_network_lifetime.png
    figures/fig_6_6_pdr.png
    figures/fig_6_7_throughput.png
    figures/fig_6_8_energy.png
"""

import argparse
import time

from hetnet_simulation import HetNetSimulation
from visualize import (
    plot_node_init,
    plot_clustering,
    plot_node_selection,
    plot_pso_performance,
    plot_comparison_charts,
)


def main(rounds: int = 6, seed: int = 42):
    print("=" * 60)
    print("5G HetNet — PSO-KM(CM2) Simulation")
    print("B.Tech Thesis | CVR College of Engineering | 2023-24")
    print("=" * 60)

    t0 = time.time()

    # --- Initialise network -----------------------------------------------
    print("\n[1/4] Initialising network (200 nodes, 200×200 m area)...")
    sim = HetNetSimulation(seed=seed)
    plot_node_init(sim)

    # --- Clustering --------------------------------------------------------
    print("\n[2/4] Running PSO-KM(CM2) clustering...")
    sim.run_clustering()
    plot_clustering(sim)
    plot_node_selection(sim)

    # --- PSO performance bar chart -----------------------------------------
    print("\n[3/4] Generating PSO performance chart...")
    plot_pso_performance()

    # --- Protocol comparison -----------------------------------------------
    print("\n[4/4] Running simulation across 6 time slots (50–300ms)...")
    pso_results = sim.run(rounds=rounds)

    print("\nPSO Results Summary:")
    print(f"{'Time(ms)':<10} {'Lifetime':<12} {'PDR%':<8} {'Throughput':<14} {'Energy(J)'}")
    print("-" * 55)
    for r in pso_results:
        print(f"{r['time_ms']:<10} {r['network_lifetime_ms']:<12} "
              f"{r['pdr']:<8} {r['throughput']:<14} {r['energy_joules']}")

    plot_comparison_charts(pso_results)

    print(f"\nDone in {time.time() - t0:.1f}s. All figures saved to ./figures/")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='5G HetNet PSO-KM(CM2) simulation')
    parser.add_argument('--rounds', type=int, default=6,
                        help='Number of 50ms time slots (default: 6)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    args = parser.parse_args()
    main(rounds=args.rounds, seed=args.seed)
