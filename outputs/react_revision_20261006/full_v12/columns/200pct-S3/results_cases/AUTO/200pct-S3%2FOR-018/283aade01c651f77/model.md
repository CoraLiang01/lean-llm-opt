#### Mathematical Model

Let $I$ be the set of distribution centers (suppliers) and $J$ the set of customer groups, as defined by the source data.

Decision variables:
- $x_{ij} \geq 0$: quantity shipped from supplier $i \in I$ to customer $j \in J$ (continuous).

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]
where $c_{ij}$ is the transportation cost per unit from supplier $i$ to customer $j$.

Subject to:
1. Demand satisfaction:
   \[
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   \]
   where $d_j$ is the demand of customer $j$.

2. Supply capacity:
   \[
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   \]
   where $s_i$ is the supply capacity of supplier $i$.

3. Non-negativity:
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]

#### Data Mapping

- $I$: supplier IDs from `supply_capacity.csv` (`file_1_view_0`, column `supplier_id`)
- $J$: customer IDs from `customer_demand.csv` (`file_0_view_0`, column `customer_id`)
- $d_j$: demand for customer $j$ from `customer_demand.csv` (`file_0_view_0`, column `demand`)
- $s_i$: supply capacity for supplier $i$ from `supply_capacity.csv` (`file_1_view_0`, column `supply_capacity`)
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer $j$ from `transportation_costs.csv` (`file_2_view_0`, row `supplier_id`, column `transportation_cost_to_{customer_id}`)

Index sets and all parameters are defined exactly as in the current source data, preserving all identifiers and coefficients.