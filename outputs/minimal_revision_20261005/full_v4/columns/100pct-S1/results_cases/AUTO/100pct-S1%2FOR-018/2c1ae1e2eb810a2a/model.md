##### Decision Variables

For each supplier $i$ (distribution center) and customer group $j$:
- $x_{ij} \geq 0$: quantity shipped from supplier $i$ to customer $j$ (continuous).

##### Parameters

- $I$: set of suppliers (distribution centers), from column `supplier_id` in `supply_capacity.csv` and `transportation_costs.csv`.
- $J$: set of customers, from column `customer_id` in `customer_demand.csv` and columns `transportation_cost_to_{customer_id}` in `transportation_costs.csv`.
- $d_j$: demand of customer $j$, from `customer_demand.csv` (`demand` column, indexed by `customer_id`).
- $s_i$: supply capacity of supplier $i$, from `supply_capacity.csv` (`supply_capacity` column, indexed by `supplier_id`).
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer $j$, from `transportation_costs.csv` (row `supplier_id`, column `transportation_cost_to_{customer_id}`).

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

- $I$ (suppliers): all `supplier_id` in `supply_capacity.csv` (`file_1_view_0`, column `supplier_id`) and `transportation_costs.csv` (`file_2_view_0`, column `supplier_id`)
- $J$ (customers): all `customer_id` in `customer_demand.csv` (`file_0_view_0`, column `customer_id`) and columns `transportation_cost_to_{customer_id}` in `transportation_costs.csv` (`file_2_view_0`)
- $d_j$: from `customer_demand.csv` (`file_0_view_0`, columns `customer_id`, `demand`)
- $s_i$: from `supply_capacity.csv` (`file_1_view_0`, columns `supplier_id`, `supply_capacity`)
- $c_{ij}$: from `transportation_costs.csv` (`file_2_view_0`, row `supplier_id`, column `transportation_cost_to_{customer_id}`)

##### Index sets and parameter values are to be taken exactly as listed in the retrieved tables, preserving all identifiers and coefficients.