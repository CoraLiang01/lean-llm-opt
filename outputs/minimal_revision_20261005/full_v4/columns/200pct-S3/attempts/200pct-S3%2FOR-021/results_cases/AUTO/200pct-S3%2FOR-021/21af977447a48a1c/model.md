##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity shipped from plant $i$ to retail outlet $j$, for all $i \in I$, $j \in J$.

##### Index Sets

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}\}$ (production plants, from supply_capacity.csv and transportation_costs.csv)
- $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}$ (retail outlets, from customer_demand.csv and transportation_costs.csv)

##### Parameters

- $d_j$: demand of outlet $j$ (from customer_demand.csv, table_id: file_0_view_0)
  - $d_{\text{C1}} = 94$
  - $d_{\text{C2}} = 39$
  - $d_{\text{C3}} = 65$
  - $d_{\text{C4}} = 435$
- $s_i$: supply capacity of plant $i$ (from supply_capacity.csv, table_id: file_1_view_0)
  - $s_{\text{S1}} = 2531$
  - $s_{\text{S2}} = 20$
  - $s_{\text{S3}} = 210$
  - $s_{\text{S4}} = 241$
- $c_{ij}$: transportation cost per unit from plant $i$ to outlet $j$ (from transportation_costs.csv, table_id: file_2_view_0)
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
2. **Supply capacity:**  
   For each plant $i \in I$,
   $$
   \sum_{j \in J} x_{ij} \leq s_i
   $$
3. **Non-negativity:**  
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

---

#### Data Mapping

- $I$ (plants): supplier_id from supply_capacity.csv (table_id: file_1_view_0) and transportation_costs.csv (table_id: file_2_view_0)
- $J$ (outlets): customer_id from customer_demand.csv (table_id: file_0_view_0) and transportation_costs.csv (table_id: file_2_view_0)
- $d_j$: demand from customer_demand.csv (table_id: file_0_view_0, column: demand)
- $s_i$: supply_capacity from supply_capacity.csv (table_id: file_1_view_0, column: supply_capacity)
- $c_{ij}$: transportation_cost_to_C* from transportation_costs.csv (table_id: file_2_view_0, columns: transportation_cost_to_C1, ..., transportation_cost_to_C4, row: supplier_id)

---

**All indices, parameters, and coefficients are bound exactly to the retrieved data and identifiers.**