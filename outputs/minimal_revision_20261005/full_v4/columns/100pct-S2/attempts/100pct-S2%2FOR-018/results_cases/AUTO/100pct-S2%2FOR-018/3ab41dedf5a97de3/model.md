##### Decision Variables

For each supplier $i$ in $I$ and customer $j$ in $J$:
- $x_{ij} \geq 0$: quantity shipped from supplier $i$ to customer $j$ (continuous).

##### Parameters

- $I$: set of suppliers (distribution centers), from `supply_capacity.csv` and `transportation_costs.csv`:
  $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}, \text{S9}, \text{S10}, \text{S11}, \text{S12}\}$
- $J$: set of customers, from `customer_demand.csv` and `transportation_costs.csv`:
  $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}, \text{C11}, \text{C12}\}$
- $d_j$: demand of customer $j$, from `customer_demand.csv` (table_id: file_0_view_0, columns: customer_id, demand)
- $s_i$: supply capacity of supplier $i$, from `supply_capacity.csv` (table_id: file_1_view_0, columns: supplier_id, supply_capacity)
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer $j$, from `transportation_costs.csv` (table_id: file_2_view_0, columns: supplier_id, transportation_cost_to_C1, ..., transportation_cost_to_C12)

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction** (each customer receives at least its demand):
   $$
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   $$
2. **Supply capacity** (each supplier does not exceed its capacity):
   $$
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   $$
3. **Non-negativity**:
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

---

#### Data Mapping

- $I$ (suppliers): all `supplier_id` in `supply_capacity.csv` (table_id: file_1_view_0) and `transportation_costs.csv` (table_id: file_2_view_0)
- $J$ (customers): all `customer_id` in `customer_demand.csv` (table_id: file_0_view_0) and columns `transportation_cost_to_C*` in `transportation_costs.csv` (table_id: file_2_view_0)
- $d_j$: `demand` for each `customer_id` in `customer_demand.csv` (table_id: file_0_view_0)
- $s_i$: `supply_capacity` for each `supplier_id` in `supply_capacity.csv` (table_id: file_1_view_0)
- $c_{ij}$: value in `transportation_costs.csv` (table_id: file_2_view_0), row `supplier_id` $i$, column `transportation_cost_to_{j}$

---

**All sets, parameters, and coefficients are defined exactly as retrieved from the source files and table_ids above.**