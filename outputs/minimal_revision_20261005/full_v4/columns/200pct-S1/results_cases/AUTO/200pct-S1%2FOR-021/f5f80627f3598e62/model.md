##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity shipped from plant $i$ to retail outlet $j$, for all $i \in I$, $j \in J$.

##### Index Sets

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}\}$ (production plants, from `supply_capacity.csv` and `transportation_costs.csv`)
- $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}$ (retail outlets, from `customer_demand.csv` and `transportation_costs.csv`)

##### Parameters

- $d_j$: demand of outlet $j$ (from `customer_demand.csv`)
  - $d_{\text{C1}} = 94$
  - $d_{\text{C2}} = 39$
  - $d_{\text{C3}} = 65$
  - $d_{\text{C4}} = 435$
- $s_i$: supply capacity of plant $i$ (from `supply_capacity.csv`)
  - $s_{\text{S1}} = 2531$
  - $s_{\text{S2}} = 20$
  - $s_{\text{S3}} = 210$
  - $s_{\text{S4}} = 241$
- $c_{ij}$: transportation cost per unit from plant $i$ to outlet $j$ (from `transportation_costs.csv`)

| $c_{ij}$         | C1                  | C2                  | C3                  | C4                  |
|------------------|---------------------|---------------------|---------------------|---------------------|
| S1               | 543.756480860856    | 23.685276141764653  | 23.676386730773032  | 447.75143678673766  |
| S2               | 883.9151090405642   | 0.04977684765576961 | 0.0350986687216299  | 44.45588531711622   |
| S3               | 537.3456896658107   | 23.769274659075112  | 498.95659249465467  | 440.60737890439776  |
| S4               | 1791.493192397229   | 68.21633865655126   | 1432.4837339656747  | 1527.7635425462734  |

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction:**  
   For each outlet $j \in J$,
   $$
   \sum_{i \in I} x_{ij} \geq d_j
   $$
   - $j = \text{C1}: \sum_{i} x_{i,\text{C1}} \geq 94$
   - $j = \text{C2}: \sum_{i} x_{i,\text{C2}} \geq 39$
   - $j = \text{C3}: \sum_{i} x_{i,\text{C3}} \geq 65$
   - $j = \text{C4}: \sum_{i} x_{i,\text{C4}} \geq 435$

2. **Supply capacity:**  
   For each plant $i \in I$,
   $$
   \sum_{j \in J} x_{ij} \leq s_i
   $$
   - $i = \text{S1}: \sum_{j} x_{\text{S1},j} \leq 2531$
   - $i = \text{S2}: \sum_{j} x_{\text{S2},j} \leq 20$
   - $i = \text{S3}: \sum_{j} x_{\text{S3},j} \leq 210$
   - $i = \text{S4}: \sum_{j} x_{\text{S4},j} \leq 241$

3. **Non-negativity:**  
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

---

#### Data Mapping

- $I$ (plants): `file_1_view_0.supplier_id` and `file_2_view_0.supplier_id`
- $J$ (outlets): `file_0_view_0.customer_id` and `file_2_view_0` columns `transportation_cost_to_C1`, ..., `transportation_cost_to_C4`
- $d_j$: `file_0_view_0.demand`
- $s_i$: `file_1_view_0.supply_capacity`
- $c_{ij}$: `file_2_view_0` with row `supplier_id = i`, column `transportation_cost_to_{j}`

---

**Complete Model:**

Minimize
$$
\sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

subject to
$$
\sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
$$
$$
\sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
$$
$$
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
$$

with all parameters and sets mapped as above.