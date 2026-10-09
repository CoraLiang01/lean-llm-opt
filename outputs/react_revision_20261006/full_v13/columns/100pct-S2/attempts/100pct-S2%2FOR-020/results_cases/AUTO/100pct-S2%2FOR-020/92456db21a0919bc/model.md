##### Mathematical Model

Let $I$ be the set of warehouses (indexed by $i$), and $J$ the set of stores (indexed by $j$):

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$
- $J = \{\text{D1}, \text{D2}, \text{D3}, \text{D4}, \text{D5}\}$

Let:
- $x_{ij} \geq 0$: quantity shipped from warehouse $i \in I$ to store $j \in J$ (continuous variable)
- $c_{ij}$: unit transportation cost from warehouse $i$ to store $j$
- $d_j$: demand of store $j$
- $s_i$: supply capacity of warehouse $i$

**Objective:**
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

**Subject to:**

1. **Demand satisfaction (for each store):**
   \[
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   \]

2. **Supply capacity (for each warehouse):**
   \[
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   \]

3. **Non-negativity:**
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]

##### Data Mapping

- $I$ (warehouses): All unique values in column `supplier_id` of table_id `file_1_view_0` (from `supply_capacity.csv`)
- $J$ (stores): All unique values in column `customer_id` of table_id `file_0_view_0` (from `customer_demand.csv`)
- $d_j$: Value in column `demand_units` for store $j$ in table_id `file_0_view_0`
- $s_i$: Value in column `supply_capacity_units` for warehouse $i$ in table_id `file_1_view_0`
- $c_{ij}$: Value in column `transportation_cost_to_{j}` for warehouse $i$ in table_id `file_2_view_0`, where `{j}$ is the store ID (e.g., `D1`, `D2`, etc.)

All index sets, parameters, and coefficients are defined exactly as in the current CSV data, preserving all identifiers and source order.