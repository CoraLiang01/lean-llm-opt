##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from plant $i \in I$ to retail outlet $j \in J$ (continuous).

##### Objective Function

$\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}$

##### Constraints

1. Demand satisfaction: $\sum_{i \in I} x_{ij} \geq d_j,\quad \forall j \in J$
2. Supply capacity: $\sum_{j \in J} x_{ij} \leq s_i,\quad \forall i \in I$
3. Non-negativity: $x_{ij} \geq 0$ (continuous)

Where:

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}\}$ (plants)
- $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}$ (retail outlets)

##### Data Mapping

- Demand $d_j$ (from customer_demand.csv, column "demand"):
  - $d_{\text{C1}} = 94$
  - $d_{\text{C2}} = 39$
  - $d_{\text{C3}} = 65$
  - $d_{\text{C4}} = 435$

- Supply capacity $s_i$ (from supply_capacity.csv, column "supply_capacity"):
  - $s_{\text{S1}} = 2531$
  - $s_{\text{S2}} = 20$
  - $s_{\text{S3}} = 210$
  - $s_{\text{S4}} = 241$

- Transportation cost $c_{ij}$ (from transportation_costs.csv, row for $i$, column for $j$):
  - $c_{\text{S1},\text{C1}} = 543.756480860856$
  - $c_{\text{S1},\text{C2}} = 23.685276141764653$
  - $c_{\text{S1},\text{C3}} = 23.676386730773032$
  - $c_{\text{S1},\text{C4}} = 447.75143678673766$
  - $c_{\text{S2},\text{C1}} = 883.9151090405642$
  - $c_{\text{S2},\text{C2}} = 0.04977684765576961$
  - $c_{\text{S2},\text{C3}} = 0.0350986687216299$
  - $c_{\text{S2},\text{C4}} = 44.45588531711622$
  - $c_{\text{S3},\text{C1}} = 537.3456896658107$
  - $c_{\text{S3},\text{C2}} = 23.769274659075112$
  - $c_{\text{S3},\text{C3}} = 498.95659249465467$
  - $c_{\text{S3},\text{C4}} = 440.60737890439776$
  - $c_{\text{S4},\text{C1}} = 1791.493192397229$
  - $c_{\text{S4},\text{C2}} = 68.21633865655126$
  - $c_{\text{S4},\text{C3}} = 1432.4837339656747$
  - $c_{\text{S4},\text{C4}} = 1527.7635425462734$

- $x_{ij}$ is defined for all $i \in I$, $j \in J$.

All data is mapped directly from the supplied CSV columns and rows, preserving all identifiers and coefficients.