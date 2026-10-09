#### Mathematical Model

Let $S$ be the set of warehouses (indexed by $i$), and $D$ the set of stores (indexed by $j$):

- $S = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$
- $D = \{\text{D1}, \text{D2}, \text{D3}, \text{D4}, \text{D5}\}$

Let:
- $x_{ij} \geq 0$: quantity shipped from warehouse $i \in S$ to store $j \in D$ (continuous variable)
- $c_{ij}$: unit transportation cost from warehouse $i$ to store $j$
- $d_j$: demand of store $j$
- $s_i$: supply capacity of warehouse $i$

**Objective:**
\[
\min \sum_{i \in S} \sum_{j \in D} c_{ij} x_{ij}
\]

**Subject to:**

1. **Demand satisfaction (each store's demand must be met):**
   \[
   \sum_{i \in S} x_{ij} \geq d_j \qquad \forall j \in D
   \]

2. **Supply capacity (each warehouse's shipments cannot exceed its capacity):**
   \[
   \sum_{j \in D} x_{ij} \leq s_i \qquad \forall i \in S
   \]

3. **Non-negativity:**
   \[
   x_{ij} \geq 0 \qquad \forall i \in S,\, j \in D
   \]

#### Data Mapping

- Sets:
    - $S$ (warehouses): All unique values in column `supplier_id` of table_id `file_1_view_0` and `file_2_view_0`
    - $D$ (stores): All unique values in column `customer_id` of table_id `file_0_view_0`
- Parameters:
    - $d_j$: For each $j \in D$, value from column `demand_units` in table_id `file_0_view_0` where `customer_id` = $j$
    - $s_i$: For each $i \in S$, value from column `supply_capacity_units` in table_id `file_1_view_0` where `supplier_id` = $i$
    - $c_{ij}$: For each $i \in S$, $j \in D$, value from column `transportation_cost_to_{j}` in table_id `file_2_view_0` where `supplier_id` = $i$ and $j$ is the store ID (e.g., `transportation_cost_to_D1` for $j = \text{D1}$)
- Variables:
    - $x_{ij}$: Decision variable for each $i \in S$, $j \in D$

All indices, parameters, and coefficients are to be mapped exactly as above from the current data.