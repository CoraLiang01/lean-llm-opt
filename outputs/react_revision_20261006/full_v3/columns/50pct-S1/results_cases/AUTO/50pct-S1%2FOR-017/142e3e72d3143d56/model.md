##### Decision Variables

For each supplier $i$ in the set $I$ (from "supply_capacity.csv") and each customer group $j$ in the set $J$ (from "customer_demand.csv"), let
$$
x_{ij} \geq 0
$$
be the continuous quantity shipped from supplier $i$ to customer group $j$.

##### Objective Function

Minimize the total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$
where $c_{ij}$ is the transportation cost per unit from supplier $i$ to customer $j$ (from "transportation_costs.csv").

##### Constraints

1. **Demand satisfaction:** For each customer group $j \in J$,
$$
\sum_{i \in I} x_{ij} \geq d_j
$$
where $d_j$ is the demand of customer $j$ (from "customer_demand.csv").

2. **Supply capacity:** For each supplier $i \in I$,
$$
\sum_{j \in J} x_{ij} \leq s_i
$$
where $s_i$ is the supply capacity of supplier $i$ (from "supply_capacity.csv").

3. **Non-negativity:** For all $i \in I$, $j \in J$,
$$
x_{ij} \geq 0
$$

##### Data Mapping

- $I$ (suppliers): All unique `supplier_id` values from "supply_capacity.csv" and "transportation_costs.csv" (table_id: file_1_view_0 and file_2_view_0, column: supplier_id).
- $J$ (customers): All unique `customer_id` values from "customer_demand.csv" and columns in "transportation_costs.csv" (table_id: file_0_view_0, column: customer_id; file_2_view_0, columns: transportation_cost_to_C1, ..., transportation_cost_to_C10).
- $d_j$: Demand for customer $j$ from "customer_demand.csv" (table_id: file_0_view_0, column: demand, indexed by customer_id).
- $s_i$: Supply capacity for supplier $i$ from "supply_capacity.csv" (table_id: file_1_view_0, column: supply_capacity, indexed by supplier_id).
- $c_{ij}$: Transportation cost per unit from supplier $i$ to customer $j$ from "transportation_costs.csv" (table_id: file_2_view_0, row: supplier_id, column: transportation_cost_to_Ck where $j$ corresponds to $Ck$).

Index sets and all coefficients are defined by the full returned records in the above tables.