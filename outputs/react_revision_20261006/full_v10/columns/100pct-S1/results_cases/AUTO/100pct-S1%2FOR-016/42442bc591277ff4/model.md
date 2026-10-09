Mathematical Optimization Model

Decision Variables

For each distribution center (supplier) $i$ and customer group $j$,
$$
x_{ij} \geq 0
$$
is the continuous quantity shipped from supplier $i$ to customer $j$.

Index Sets

- $I$: set of suppliers (distribution centers), from column "supplier_id" in file_1_view_0 and file_2_view_0.
- $J$: set of customers, from column "customer_id" in file_0_view_0 and as suffixes in file_2_view_0 columns.

Parameters

- $d_j$: demand of customer $j$, from "demand_units" in file_0_view_0.
- $s_i$: supply capacity of supplier $i$, from "supply_capacity_units" in file_1_view_0.
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer $j$, from "transportation_cost_to_Ck" in file_2_view_0.

Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

Constraints

1. Demand satisfaction for each customer:
$$
\sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
$$

2. Supply capacity for each supplier:
$$
\sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
$$

3. Non-negativity:
$$
x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
$$

Data Mapping

- $I$: All "supplier_id" in file_1_view_0 and file_2_view_0.
- $J$: All "customer_id" in file_0_view_0 and as suffixes in file_2_view_0 columns (e.g., "transportation_cost_to_C1" $\rightarrow$ "C1").
- $d_j$: file_0_view_0, column "demand_units", row with "customer_id" = $j$.
- $s_i$: file_1_view_0, column "supply_capacity_units", row with "supplier_id" = $i$.
- $c_{ij}$: file_2_view_0, row with "supplier_id" = $i$, column "transportation_cost_to_Ck" where $k = j$.

All index sets, parameters, and constraints are defined exactly as in the current data. No data is omitted or aggregated. Variable domains and all bounds are as specified in the user query and source data.