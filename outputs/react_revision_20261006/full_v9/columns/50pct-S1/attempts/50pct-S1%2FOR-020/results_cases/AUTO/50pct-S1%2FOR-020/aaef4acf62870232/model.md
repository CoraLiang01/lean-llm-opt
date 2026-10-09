##### Mathematical Model

Let $I$ be the set of warehouses (indexed by $i$), and $J$ the set of stores (indexed by $j$):

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$
- $J = \{\text{D1}, \text{D2}, \text{D3}, \text{D4}, \text{D5}\}$

Let:
- $x_{ij} \geq 0$: quantity shipped from warehouse $i$ to store $j$ (continuous variable)
- $d_j$: demand of store $j$
- $s_i$: supply capacity of warehouse $i$
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$

**Objective:**
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

**Subject to:**

1. **Demand satisfaction (each store's demand must be met):**
   \[
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   \]

2. **Supply capacity (each warehouse's shipments cannot exceed its capacity):**
   \[
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   \]

3. **Non-negativity:**
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]

##### Data Mapping

- $I$ (warehouses): All unique values in column `supplier_id` of `supply_capacity.csv` (`file_1_view_0`)
- $J$ (stores): All unique values in column `customer_id` of `customer_demand.csv` (`file_0_view_0`)
- $d_j$: Value in column `demand_units` for store $j$ in `customer_demand.csv` (`file_0_view_0`)
- $s_i$: Value in column `supply_capacity_units` for warehouse $i$ in `supply_capacity.csv` (`file_1_view_0`)
- $c_{ij}$: Value in column `transportation_cost_to_{j}` for warehouse $i$ in `transportation_costs.csv` (`file_2_view_0`), where `{j}$ is the store ID (e.g., `D1`, `D2`, etc.)

- Decision variables $x_{ij}$ are defined for all $i \in I$, $j \in J$.

All index sets, parameters, and constraints are mapped directly from the current CSV data as described above. No data is omitted or aggregated.