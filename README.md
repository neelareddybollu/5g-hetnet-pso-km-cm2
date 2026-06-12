# 5G HetNet Optimisation — PSO-KM(CM2)

**Optimal User Association, Backhaul Routing and Switching Off in 5G HetNets with Mesh mm-Wave Backhaul Links**

> B.Tech Final Year Project — Electronics & Communication Engineering  
> CVR College of Engineering (JNTUH) | 2023–24  
> Contributors: E. Abhilash, **B. Neela Reddy**, K. Vijay Varun Tej  
> Supervisor: Mr. L. Manjunath, Associate Professor

---

## Overview

This repository contains a Python reconstruction of the simulation and algorithm from the above thesis. The original implementation was in MATLAB/Simulink; this version reproduces all result figures in Python.

The core problem: in a **5G Heterogeneous Network (HetNet)**, small cells (SCs) need to connect back to the core network via **millimetre-wave (mm-Wave) mesh backhaul links**. Three things must be jointly optimised:

1. **User Association** — which small cell should each user connect to?
2. **Backhaul Routing** — how does data flow across the mesh links?
3. **Switching Off** — which idle SCs can be powered down to save energy?

---

## Algorithm: PSO-KM(CM2)

The key contribution is an enhanced version of PSO-based K-Means clustering (PSO-KM):

### Standard PSO-KM
- Each "particle" = a candidate set of K cluster centroids
- Velocity update (eq. 2 from thesis):

```
V_{i,j,n+1} = w·V_{i,j,n} + c1·r1·(P_{i,j,n} - X_{i,j,n}) + c2·r2·(G - X_{i,j,n})
```

- One K-Means iteration per PSO step to exploit local solution quality
- Fitness = MSE (eq. 4): lower is better

### CM2 — Enhanced Cluster Matching (novel contribution)
The problem with standard PSO-KM: the *ordering* of cluster centroids in a particle is arbitrary, causing divergence when particles share information during velocity updates.

**Fix (CM2):** Before each velocity update, rearrange each particle's centroids to match the ordering of the global best particle using minimum-distance greedy pairing.

```
Distance matrix → greedy min-distance pairing → reorder centroids → PSO update
```

This produces tighter clusters and faster convergence vs. K-Means, PSO-KM, PK-Means, and PSO-KM(CM).

### Parameters (from Table 1 of thesis)

| Parameter | Value |
|-----------|-------|
| Swarm size | 6 |
| Inertia weight (w) | 0.7 |
| Cognitive factor (c1) | 0.2 |
| Social factor (c2) | 0.2 |
| Max iterations | 40 |

---

## Results

PSO outperforms OEERP, LEACH, DRINA, and BCDCA across all metrics:

| Metric (at 50ms) | PSO | OEERP | LEACH | DRINA | BCDCA |
|---|---|---|---|---|---|
| Network Lifetime (ms) | **11,000** | 7,000 | 2,500 | 8,300 | 4,900 |
| Packet Delivery Ratio (%) | **100** | 60 | 64 | 90 | 20 |
| Throughput (×10⁴ bps) | **7.8** | 5.5 | 5.7 | 7.4 | 1.6 |
| Energy Consumption (J) | 8 | 10 | 20 | **5** | 12 |

---

## File Structure

```
├── pso_km_cm2.py          # Core PSO-KM(CM2) algorithm
├── hetnet_simulation.py   # 5G HetNet network model + baseline protocols
├── visualize.py           # Reproduces thesis Figs 6.1–6.8
├── main.py                # Entry point
├── requirements.txt
└── figures/               # Generated on first run
    ├── fig_6_1_node_init.png
    ├── fig_6_2_clustering.png
    ├── fig_6_3_node_selection.png
    ├── fig_6_4_pso_performance.png
    ├── fig_6_5_network_lifetime.png
    ├── fig_6_6_pdr.png
    ├── fig_6_7_throughput.png
    └── fig_6_8_energy.png
```

---

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run simulation and generate all figures
python main.py

# Custom options
python main.py --rounds 6 --seed 42
```

---

## Technologies

`Python` `NumPy` `SciPy` `Matplotlib` | Algorithms: `PSO` `K-Means` `CM2`

Domain: `5G NR` `HetNet` `mm-Wave Backhaul` `Network Optimisation` `Wireless Sensor Networks`
