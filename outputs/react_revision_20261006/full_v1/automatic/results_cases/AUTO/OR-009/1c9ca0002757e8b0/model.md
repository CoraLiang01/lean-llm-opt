##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity shipped from plant $i$ to retail outlet $j$, for all $i \in I$, $j \in J$.

Where:
- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}\}$ (plants)
- $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}$ (retail outlets)

##### Objective Function

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction** (each retail outlet receives at least its demand):
   $$
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   $$
2. **Supply capacity** (each plant does not ship more than its capacity):
   $$
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   $$
3. **Non-negativity**:
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

##### Data Mapping

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}\}$ (from supply_capacity.csv, column "Unnamed: 0", table_id: file_1_view_0)
- $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}$ (from customer_demand.csv, column "customer", table_id: file_0_view_0)
- $d_j$ (demand for outlet $j$):
  - $d_{\text{C1}} = 94$
  - $d_{\text{C2}} = 39$
  - $d_{\text{C3}} = 65$
  - $d_{\text{C4}} = 435$
  (from customer_demand.csv, column "demand", table_id: file_0_view_0)
- $s_i$ (supply capacity for plant $i$):
  - $s_{\text{S1}} = 2531$
  - $s_{\text{S2}} = 20$
  - $s_{\text{S3}} = 210$
  - $s_{\text{S4}} = 241$
  (from supply_capacity.csv, column "supply_capacity", table_id: file_1_view_0)
- $c_{ij}$ (transportation cost per unit from plant $i$ to outlet $j$) from transportation_costs.csv, table_id: file_2_view_0:

|        | C1              | C2                | C3                | C4                |
|--------|-----------------|-------------------|-------------------|-------------------|
| S1     | 543.756480860856 | 23.685276141764653 | 23.676386730773032 | 447.75143678673766 |
| S2     | 883.9151090405642 | 0.04977684765576961 | 0.0350986687216299 | 44.45588531711622  |
| S3     | 537.3456896658107 | 23.769274659075112 | 498.95659249465467 | 440.60737890439776 |
| S4     | 1791.493192397229 | 68.21633865655126  | 1432.4837339656747 | 1527.7635425462734 |

- $c_{ij}$ is from transportation_costs.csv, columns "C1", "C2", "C3", "C4", rows "S1", "S2", "S3", "S4", table_id: file_2_view_0.

##### Complete Model

Minimize
$$
543.756480860856\,x_{\text{S1},\text{C1}} + 23.685276141764653\,x_{\text{S1},\text{C2}} + 23.676386730773032\,x_{\text{S1},\text{C3}} + 447.75143678673766\,x_{\text{S1},\text{C4}} \\
+ 883.9151090405642\,x_{\text{S2},\text{C1}} + 0.04977684765576961\,x_{\text{S2},\text{C2}} + 0.0350986687216299\,x_{\text{S2},\text{C3}} + 44.45588531711622\,x_{\text{S2},\text{C4}} \\
+ 537.3456896658107\,x_{\text{S3},\text{C1}} + 23.769274659075112\,x_{\text{S3},\text{C2}} + 498.95659249465467\,x_{\text{S3},\text{C3}} + 440.60737890439776\,x_{\text{S3},\text{C4}} \\
+ 1791.493192397229\,x_{\text{S4},\text{C1}} + 68.21633865655126\,x_{\text{S4},\text{C2}} + 1432.4837339656747\,x_{\text{S4},\text{C3}} + 1527.7635425462734\,x_{\text{S4},\text{C4}}
$$

Subject to:
$$
x_{\text{S1},\text{C1}} + x_{\text{S2},\text{C1}} + x_{\text{S3},\text{C1}} + x_{\text{S4},\text{C1}} \geq 94 \\
x_{\text{S1},\text{C2}} + x_{\text{S2},\text{C2}} + x_{\text{S3},\text{C2}} + x_{\text{S4},\text{C2}} \geq 39 \\
x_{\text{S1},\text{C3}} + x_{\text{S2},\text{C3}} + x_{\text{S3},\text{C3}} + x_{\text{S4},\text{C3}} \geq 65 \\
x_{\text{S1},\text{C4}} + x_{\text{S2},\text{C4}} + x_{\text{S3},\text{C4}} + x_{\text{S4},\text{C4}} \geq 435 \\
$$

$$
x_{\text{S1},\text{C1}} + x_{\text{S1},\text{C2}} + x_{\text{S1},\text{C3}} + x_{\text{S1},\text{C4}} \leq 2531 \\
x_{\text{S2},\text{C1}} + x_{\text{S2},\text{C2}} + x_{\text{S2},\text{C3}} + x_{\text{S2},\text{C4}} \leq 20 \\
x_{\text{S3},\text{C1}} + x_{\text{S3},\text{C2}} + x_{\text{S3},\text{C3}} + x_{\text{S3},\text{C4}} \leq 210 \\
x_{\text{S4},\text{C1}} + x_{\text{S4},\text{C2}} + x_{\text{S4},\text{C3}} + x_{\text{S4},\text{C4}} \leq 241 \\
$$

$$
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
$$

##### Data Mapping

- $I$ (plants): file_1_view_0, column "Unnamed: 0"
- $J$ (retail outlets): file_0_view_0, column "customer"
- $d_j$: file_0_view_0, column "demand"
- $s_i$: file_1_view_0, column "supply_capacity"
- $c_{ij}$: file_2_view_0, columns "C1", "C2", "C3", "C4", rows "S1", "S2", "S3", "S4"