##### Sets

- Plants (Supply nodes): $S = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}\}$
- Customers (Demand nodes): $C = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}$

##### Parameters

- Customer demands:
  - $d_{\text{C1}} = 94$
  - $d_{\text{C2}} = 39$
  - $d_{\text{C3}} = 65$
  - $d_{\text{C4}} = 435$

- Plant supply capacities:
  - $u_{\text{S1}} = 2531$
  - $u_{\text{S2}} = 20$
  - $u_{\text{S3}} = 210$
  - $u_{\text{S4}} = 241$

- Transportation costs per unit ($c_{ij}$, from plant $i$ to customer $j$):

\[
\begin{array}{c|cccc}
 & \text{C1} & \text{C2} & \text{C3} & \text{C4} \\
\hline
\text{S1} & 543.756480860856 & 23.685276141764653 & 23.676386730773032 & 447.75143678673766 \\
\text{S2} & 883.9151090405642 & 0.04977684765576961 & 0.0350986687216299 & 44.45588531711622 \\
\text{S3} & 537.3456896658107 & 23.769274659075112 & 498.95659249465467 & 440.60737890439776 \\
\text{S4} & 1791.493192397229 & 68.21633865655126 & 1432.4837339656747 & 1527.7635425462734 \\
\end{array}
\]

##### Decision Variables

- $x_{ij} \geq 0$: Number of units shipped from plant $i \in S$ to customer $j \in C$

##### Objective Function

\[
\min \sum_{i \in S} \sum_{j \in C} c_{ij} x_{ij}
\]

##### Constraints

1. **Demand satisfaction (all customer demands must be met):**

\[
\sum_{i \in S} x_{ij} = d_j \quad \forall j \in C
\]

That is,
\[
\sum_{i \in S} x_{i,\text{C1}} = 94
\]
\[
\sum_{i \in S} x_{i,\text{C2}} = 39
\]
\[
\sum_{i \in S} x_{i,\text{C3}} = 65
\]
\[
\sum_{i \in S} x_{i,\text{C4}} = 435
\]

2. **Supply capacity (no plant exceeds its capacity):**

\[
\sum_{j \in C} x_{ij} \leq u_i \quad \forall i \in S
\]

That is,
\[
\sum_{j \in C} x_{\text{S1},j} \leq 2531
\]
\[
\sum_{j \in C} x_{\text{S2},j} \leq 20
\]
\[
\sum_{j \in C} x_{\text{S3},j} \leq 210
\]
\[
\sum_{j \in C} x_{\text{S4},j} \leq 241
\]

3. **Non-negativity:**

\[
x_{ij} \geq 0 \quad \forall i \in S,\, j \in C
\]

##### Retrieved Information

{
  "customer_demand": {
    "C1": 94,
    "C2": 39,
    "C3": 65,
    "C4": 435
  },
  "supply_capacity": {
    "S1": 2531,
    "S2": 20,
    "S3": 210,
    "S4": 241
  },
  "transportation_costs": {
    "S1": {"C1": 543.756480860856, "C2": 23.685276141764653, "C3": 23.676386730773032, "C4": 447.75143678673766},
    "S2": {"C1": 883.9151090405642, "C2": 0.04977684765576961, "C3": 0.0350986687216299, "C4": 44.45588531711622},
    "S3": {"C1": 537.3456896658107, "C2": 23.769274659075112, "C3": 498.95659249465467, "C4": 440.60737890439776},
    "S4": {"C1": 1791.493192397229, "C2": 68.21633865655126, "C3": 1432.4837339656747, "C4": 1527.7635425462734}
  },
  "plants": ["S1", "S2", "S3", "S4"],
  "customers": ["C1", "C2", "C3", "C4"]
}