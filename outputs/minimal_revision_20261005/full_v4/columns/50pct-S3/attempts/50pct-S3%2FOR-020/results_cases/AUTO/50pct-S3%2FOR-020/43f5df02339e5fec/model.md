##### Decision Variables

Let $x_{ij} \geq 0$ be the quantity shipped from warehouse $i$ to store $j$, for all $i \in I$, $j \in J$.

##### Index Sets

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}\}$ (warehouses, from `file_1_view_0.supplier_id`)
- $J = \{\text{D1}, \text{D2}, \text{D3}, \text{D4}, \text{D5}\}$ (stores, from `file_0_view_0.customer_id`)

##### Parameters

- $d_j$: demand at store $j$ (from `file_0_view_0.demand_units`)
- $s_i$: supply capacity at warehouse $i$ (from `file_1_view_0.supply_capacity_units`)
- $c_{ij}$: transportation cost per unit from warehouse $i$ to store $j$ (from `file_2_view_0`, see Data Mapping)

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

- $I$ (warehouses): all `supplier_id` in `file_1_view_0`
- $J$ (stores): all `customer_id` in `file_0_view_0`
- $d_j$: `file_0_view_0.demand_units` for store $j$
- $s_i$: `file_1_view_0.supply_capacity_units` for warehouse $i$
- $c_{ij}$: `file_2_view_0` row with `supplier_id` $i$, column `transportation_cost_to_{j}`

##### Example parameter binding (do not enumerate, for mapping only):

- $d_{\text{D1}}$ = value in `file_0_view_0` where `customer_id` = D1, column `demand_units`
- $s_{\text{S1}}$ = value in `file_1_view_0` where `supplier_id` = S1, column `supply_capacity_units`
- $c_{\text{S1},\text{D1}}$ = value in `file_2_view_0` where `supplier_id` = S1, column `transportation_cost_to_D1`

All indices, parameters, and coefficients are bound exactly as above.