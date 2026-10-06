##### Decision Variables

For each supplier $i$ (distribution center) and customer group $j$:
- $x_{ij} \geq 0$: quantity shipped from supplier $i$ to customer $j$ (continuous).

##### Parameters

- $I$: set of suppliers (distribution centers), from column `supplier_id` in `supply_capacity.csv` and `transportation_costs.csv`.
- $J$: set of customers, from column `customer_id` in `customer_demand.csv` and columns `transportation_cost_to_{customer_id}` in `transportation_costs.csv`.
- $d_j$: demand of customer $j$, from column `demand` in `customer_demand.csv`.
- $s_i$: supply capacity of supplier $i$, from column `supply_capacity` in `supply_capacity.csv`.
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer $j$, from column `transportation_cost_to_{j}` in `transportation_costs.csv` for row `supplier_id = i`.

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction:**  
   For each customer $j \in J$,
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

- $I$ (suppliers): all `supplier_id` in `supply_capacity.csv` (`file_1_view_0`) and `transportation_costs.csv` (`file_2_view_0`)
- $J$ (customers): all `customer_id` in `customer_demand.csv` (`file_0_view_0`) and columns `transportation_cost_to_{customer_id}` in `transportation_costs.csv` (`file_2_view_0`)
- $d_j$: from `demand` column in `customer_demand.csv` (`file_0_view_0`), indexed by `customer_id`
- $s_i$: from `supply_capacity` column in `supply_capacity.csv` (`file_1_view_0`), indexed by `supplier_id`
- $c_{ij}$: from `transportation_cost_to_{customer_id}` column in `transportation_costs.csv` (`file_2_view_0`), for row `supplier_id = i` and column for $j$

##### Table IDs and Columns

- `file_0_view_0`: customer_demand.csv (`customer_id`, `demand`)
- `file_1_view_0`: supply_capacity.csv (`supplier_id`, `supply_capacity`)
- `file_2_view_0`: transportation_costs.csv (`supplier_id`, `transportation_cost_to_C1`, ..., `transportation_cost_to_C12`)

##### Index Sets

- $I = \{$S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12$\}$
- $J = \{$C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12$\}$

##### Variable Domains

- $x_{ij} \geq 0$ and continuous, for all $i \in I$, $j \in J$.

---

**This model uses all identifiers and coefficients as retrieved, with no aggregation or omission.**