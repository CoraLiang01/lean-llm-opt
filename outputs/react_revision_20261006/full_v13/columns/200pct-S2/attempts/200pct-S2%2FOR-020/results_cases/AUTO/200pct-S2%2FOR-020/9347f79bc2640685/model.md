##### Mathematical Model

Let $I$ be the set of warehouses (indexed by $i$), and $J$ the set of stores (indexed by $j$):

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$
- $J = \{\text{D1}, \text{D2}, \text{D3}, \text{D4}, \text{D5}\}$

Let $x_{ij} \geq 0$ be the quantity shipped from warehouse $i$ to store $j$ (continuous).

Parameters:
- $d_j$: demand (units) for store $j$
- $s_i$: supply capacity (units) for warehouse $i$
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:
1. **Demand satisfaction** (each store receives at least its demand):
   \[
   \sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
   \]
2. **Supply capacity** (each warehouse ships no more than its capacity):
   \[
   \sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
   \]
3. **Non-negativity**:
   \[
   x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   \]

##### Data Mapping

- $I$ (warehouses): All unique values of `supplier_id` from `file_1_view_0` (supply_capacity.csv), in source order.
- $J$ (stores): All unique values of `customer_id` from `file_0_view_0` (customer_demand.csv), in source order.
- $d_j$: For each $j \in J$, $d_j$ is the value of `demand_units` from `file_0_view_0` where `customer_id` = $j$.
- $s_i$: For each $i \in I$, $s_i$ is the value of `supply_capacity_units` from `file_1_view_0` where `supplier_id` = $i$.
- $c_{ij}$: For each $i \in I$, $j \in J$, $c_{ij}$ is the value of the column `transportation_cost_to_{j}` from `file_2_view_0` (transportation_costs.csv) in the row where `supplier_id` = $i$.

All index sets, parameters, and coefficients are defined exactly as above from the current source data. No data is omitted or aggregated.