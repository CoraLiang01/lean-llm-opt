##### Mathematical Model

Let $I$ be the set of distribution centers (suppliers) and $J$ the set of customer groups.

Let $x_{ij} \geq 0$ be the continuous quantity shipped from supplier $i \in I$ to customer $j \in J$.

**Objective:**
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

**Subject to:**

1. **Demand satisfaction:**  
   \[
   \sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
   \]

2. **Supply capacity:**  
   \[
   \sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
   \]

3. **Non-negativity:**  
   \[
   x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   \]

**Index sets and parameter mapping:**

- $I$ = set of supplier IDs from `supply_capacity.csv` and `transportation_costs.csv` (`supplier_id` in both, source: file_1_view_0 and file_2_view_0)
- $J$ = set of customer IDs from `customer_demand.csv` and `transportation_costs.csv` (columns `customer_id` in file_0_view_0 and `transportation_cost_to_Ck` in file_2_view_0)
- $d_j$ = demand for customer $j$ from column `demand` in `customer_demand.csv` (file_0_view_0)
- $s_i$ = supply capacity for supplier $i$ from column `supply_capacity` in `supply_capacity.csv` (file_1_view_0)
- $c_{ij}$ = transportation cost per unit from supplier $i$ to customer $j$ from column `transportation_cost_to_Ck` in `transportation_costs.csv` (file_2_view_0, row `supplier_id` $i$, column for $j$)

##### Data Mapping

- $I$: All `supplier_id` in `supply_capacity.csv` (file_1_view_0, column `supplier_id`) and `transportation_costs.csv` (file_2_view_0, column `supplier_id`)
- $J$: All `customer_id` in `customer_demand.csv` (file_0_view_0, column `customer_id`) and all columns with suffix `transportation_cost_to_Ck` in `transportation_costs.csv` (file_2_view_0)
- $d_j$: `customer_demand.csv` (file_0_view_0, columns `customer_id`, `demand`)
- $s_i$: `supply_capacity.csv` (file_1_view_0, columns `supplier_id`, `supply_capacity`)
- $c_{ij}$: `transportation_costs.csv` (file_2_view_0, row `supplier_id` $i$, column `transportation_cost_to_Ck` for $j$)

**Variable domain:**  
$x_{ij} \geq 0$ continuous, for all $i \in I$, $j \in J$.