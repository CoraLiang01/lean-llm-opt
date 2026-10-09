##### Mathematical Model

Let $I$ be the set of warehouses (indexed by $i$), and $J$ the set of stores (indexed by $j$):

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$
- $J = \{\text{D1}, \text{D2}, \text{D3}, \text{D4}, \text{D5}\}$

Let:
- $x_{ij} \geq 0$: quantity shipped from warehouse $i$ to store $j$ (continuous variable)
- $d_j$: demand at store $j$
- $s_i$: supply capacity at warehouse $i$
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$

**Objective:**
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

**Subject to:**

1. **Demand satisfaction (each store's demand must be met):**
   \[
   \sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
   \]

2. **Supply capacity (each warehouse's shipments cannot exceed its capacity):**
   \[
   \sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
   \]

3. **Non-negativity:**
   \[
   x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   \]

##### Data Mapping

- $I$ (warehouses): All unique values in column `supplier_id` of table_id `file_1_view_0` (supply_capacity.csv) and `file_2_view_0` (transportation_costs.csv)
- $J$ (stores): All unique values in column `customer_id` of table_id `file_0_view_0` (customer_demand.csv) and as suffixes in columns `transportation_cost_to_D*` of table_id `file_2_view_0` (transportation_costs.csv)
- $d_j$: For each $j \in J$, value from column `demand_units` in table_id `file_0_view_0` where `customer_id` = $j$
- $s_i$: For each $i \in I$, value from column `supply_capacity_units` in table_id `file_1_view_0` where `supplier_id` = $i$
- $c_{ij}$: For each $i \in I$, $j \in J$, value from column `transportation_cost_to_{j}` in table_id `file_2_view_0` where `supplier_id` = $i$

- Decision variables $x_{ij}$: defined for all $i \in I$, $j \in J$ as above.

All index sets, parameters, and coefficients are mapped directly from the current CSV data as described. No data is omitted or aggregated.