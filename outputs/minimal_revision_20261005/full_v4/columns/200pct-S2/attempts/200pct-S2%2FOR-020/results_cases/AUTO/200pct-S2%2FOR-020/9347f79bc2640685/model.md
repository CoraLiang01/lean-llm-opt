##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity shipped from warehouse $i$ to store $j$, for all $i \in I$, $j \in J$.

Where:
- $I$ = set of warehouses = $\{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$
- $J$ = set of stores = $\{\text{D1}, \text{D2}, \text{D3}, \text{D4}, \text{D5}\}$

##### Parameters

- $d_j$ = demand of store $j$ (from customer_demand.csv, table_id: file_0_view_0, column: demand_units)
- $s_i$ = supply capacity of warehouse $i$ (from supply_capacity.csv, table_id: file_1_view_0, column: supply_capacity_units)
- $c_{ij}$ = transportation cost per unit from warehouse $i$ to store $j$ (from transportation_costs.csv, table_id: file_2_view_0, columns: transportation_cost_to_D1, ..., transportation_cost_to_D5)

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction:** Each store must receive at least its demand.
   $$
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   $$
2. **Supply capacity:** Each warehouse cannot ship more than its capacity.
   $$
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   $$
3. **Non-negativity:**
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

---

#### Data Mapping

- $I$ (warehouses): All unique `supplier_id` in table_id: file_1_view_0 and file_2_view_0
- $J$ (stores): All unique `customer_id` in table_id: file_0_view_0 and columns with suffix in file_2_view_0
- $d_j$: For each $j \in J$, $d_j =$ value in `demand_units` where `customer_id` = $j$ (table_id: file_0_view_0)
- $s_i$: For each $i \in I$, $s_i =$ value in `supply_capacity_units` where `supplier_id` = $i$ (table_id: file_1_view_0)
- $c_{ij}$: For each $i \in I$, $j \in J$, $c_{ij} =$ value in `transportation_cost_to_{j}` where `supplier_id` = $i$ (table_id: file_2_view_0)

---

**All index sets, parameters, and coefficients are bound exactly to the retrieved data as described above.**