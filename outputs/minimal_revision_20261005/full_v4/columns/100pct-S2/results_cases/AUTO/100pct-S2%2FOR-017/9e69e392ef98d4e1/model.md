##### Decision Variables

For each supplier $i$ in $I$ and customer group $j$ in $J$:
- $x_{ij} \geq 0$: quantity shipped from supplier $i$ to customer group $j$ (continuous).

##### Objective Function

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction:**  
   For each customer group $j \in J$,
   $$
   \sum_{i \in I} x_{ij} \geq d_j
   $$
2. **Supply capacity:**  
   For each supplier $i \in I$,
   $$
   \sum_{j \in J} x_{ij} \leq s_i
   $$
3. **Non-negativity:**  
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

---

#### Data Mapping

- $I$ (Suppliers): All supplier_id in `supply_capacity.csv` (`file_1_view_0`)
- $J$ (Customer groups): All customer_id in `customer_demand.csv` (`file_0_view_0`)
- $d_j$: Demand for customer $j$ from `customer_demand.csv` (`file_0_view_0`, columns: customer_id, demand)
- $s_i$: Supply capacity for supplier $i$ from `supply_capacity.csv` (`file_1_view_0`, columns: supplier_id, supply_capacity)
- $c_{ij}$: Transportation cost per unit from supplier $i$ to customer $j$ from `transportation_costs.csv` (`file_2_view_0`, columns: supplier_id, transportation_cost_to_C1, ..., transportation_cost_to_C10`), where $c_{ij}$ is the value in the row with supplier_id $i$ and column `transportation_cost_to_{j}`.

##### Index Sets

- $I = \{$S1, S2, S3, S4, S5, S6, S7, S8, S9, S10$\}$
- $J = \{$C1, C2, C3, C4, C5, C6, C7, C8, C9, C10$\}$

##### Parameter Binding

- $d_j$ = value in `file_0_view_0` where `customer_id` = $j$, column `demand`
- $s_i$ = value in `file_1_view_0` where `supplier_id` = $i$, column `supply_capacity`
- $c_{ij}$ = value in `file_2_view_0` where `supplier_id` = $i$, column `transportation_cost_to_{j}`

---

**All parameters and sets are defined directly from the retrieved data.**