#### Mathematical Model

Let $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}\}$ (production plants) and $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}$ (retail outlets).

Decision variables:
$$
x_{ij} \geq 0 \quad \text{(continuous)}, \quad \forall i \in I,\, j \in J
$$
where $x_{ij}$ is the quantity shipped from plant $i$ to outlet $j$.

Objective:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

Subject to:
1. Demand satisfaction (each outlet receives at least its demand):
   $$
   \sum_{i \in I} x_{ij} \geq d_j, \quad \forall j \in J
   $$
2. Supply capacity (no plant exceeds its capacity):
   $$
   \sum_{j \in J} x_{ij} \leq s_i, \quad \forall i \in I
   $$
3. Non-negativity:
   $$
   x_{ij} \geq 0, \quad \forall i \in I,\, j \in J
   $$

#### Data Mapping

- $I$ (plants): supplier_id from supply_capacity.csv and transportation_costs.csv: S1, S2, S3, S4
- $J$ (outlets): customer_id from customer_demand.csv and transportation_costs.csv: C1, C2, C3, C4
- $d_j$: demand for outlet $j$ from customer_demand.csv (table_id: file_0_view_0, column: demand)
- $s_i$: supply_capacity for plant $i$ from supply_capacity.csv (table_id: file_1_view_0, column: supply_capacity)
- $c_{ij}$: transportation_cost from plant $i$ to outlet $j$ from transportation_costs.csv (table_id: file_2_view_0, columns: transportation_cost_to_C1, ..., transportation_cost_to_C4, row: supplier_id)

Index sets, parameters, and all coefficients are taken directly from the current CSV files as described above.