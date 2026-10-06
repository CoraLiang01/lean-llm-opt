##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity shipped from warehouse $i$ to store $j$, for all $i \in I$, $j \in J$.

##### Index Sets

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$ (warehouses, from `supply_capacity.csv` table_id: file_1_view_0, column: supplier_id)
- $J = \{\text{D1}, \text{D2}, \text{D3}, \text{D4}, \text{D5}\}$ (stores, from `customer_demand.csv` table_id: file_0_view_0, column: customer_id)

##### Parameters

- $d_j$: demand of store $j$ (from `customer_demand.csv`, table_id: file_0_view_0, column: demand_units)
- $s_i$: supply capacity of warehouse $i$ (from `supply_capacity.csv`, table_id: file_1_view_0, column: supply_capacity_units)
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$ (from `transportation_costs.csv`, table_id: file_2_view_0, columns: transportation_cost_to_D1, ..., transportation_cost_to_D5, row: supplier_id)

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction:**  
   For each store $j \in J$,
   $$
   \sum_{i \in I} x_{ij} \geq d_j
   $$
   (Each store receives at least its demand.)

2. **Supply capacity:**  
   For each warehouse $i \in I$,
   $$
   \sum_{j \in J} x_{ij} \leq s_i
   $$
   (No warehouse ships more than its capacity.)

3. **Non-negativity:**  
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

---

#### Data Mapping

- $I$ (warehouses): all `supplier_id` in `supply_capacity.csv` (table_id: file_1_view_0)
- $J$ (stores): all `customer_id` in `customer_demand.csv` (table_id: file_0_view_0)
- $d_j$: `demand_units` for $j$ in `customer_demand.csv` (table_id: file_0_view_0, column: demand_units)
- $s_i$: `supply_capacity_units` for $i$ in `supply_capacity.csv` (table_id: file_1_view_0, column: supply_capacity_units)
- $c_{ij}$: `transportation_cost_to_Dk` for $i$ in `transportation_costs.csv` (table_id: file_2_view_0, row: supplier_id, columns: transportation_cost_to_D1, ..., transportation_cost_to_D5, with $j$ corresponding to $Dk$)

All index sets, parameters, and coefficients are bound directly to the retrieved data as specified above.