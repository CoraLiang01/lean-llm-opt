##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity shipped from warehouse $i$ to store $j$, for all $i \in I$, $j \in J$.

##### Index Sets

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$ (warehouses, from `supply_capacity.csv` and `transportation_costs.csv`)
- $J = \{\text{D1}, \text{D2}, \text{D3}, \text{D4}, \text{D5}\}$ (stores, from `customer_demand.csv` and `transportation_costs.csv`)

##### Parameters

- $d_j$: demand of store $j$ (from `customer_demand.csv`)
- $s_i$: supply capacity of warehouse $i$ (from `supply_capacity.csv`)
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$ (from `transportation_costs.csv`)

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction:**  
   $$
   \sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
   $$
2. **Supply capacity:**  
   $$
   \sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
   $$
3. **Non-negativity:**  
   $$
   x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   $$

---

#### Data Mapping

- $I$ (warehouses): all `supplier_id` in `file_1_view_0` and `file_2_view_0`
- $J$ (stores): all `customer_id` in `file_0_view_0` and columns with suffix in `file_2_view_0`
- $d_j$: `demand_units` from `file_0_view_0` where `customer_id = j`
- $s_i$: `supply_capacity_units` from `file_1_view_0` where `supplier_id = i`
- $c_{ij}$: value in `file_2_view_0` where `supplier_id = i`, column `transportation_cost_to_{j}`

##### Table and Column Reference

- `file_0_view_0` (`customer_demand.csv`): columns `customer_id`, `demand_units`
- `file_1_view_0` (`supply_capacity.csv`): columns `supplier_id`, `supply_capacity_units`
- `file_2_view_0` (`transportation_costs.csv`): columns `supplier_id`, `transportation_cost_to_D1`, ..., `transportation_cost_to_D5`

##### Variable Domain

- $x_{ij} \geq 0$, continuous, for all $i \in I$, $j \in J$