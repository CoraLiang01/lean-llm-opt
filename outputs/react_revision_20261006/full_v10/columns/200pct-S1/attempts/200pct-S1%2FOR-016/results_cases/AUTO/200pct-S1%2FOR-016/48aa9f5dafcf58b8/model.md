Mathematical Optimization Model

Decision Variables

For each distribution center (supplier) $i$ and customer group $j$,
$$
x_{ij} \geq 0
$$
where $x_{ij}$ is the continuous quantity shipped from supplier $i$ to customer $j$.

Objective Function

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$
where $c_{ij}$ is the transportation cost per unit from supplier $i$ to customer $j$.

Constraints

1. Demand satisfaction (for each customer $j$):
$$
\sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
$$
where $d_j$ is the demand of customer $j$.

2. Supply capacity (for each supplier $i$):
$$
\sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
$$
where $s_i$ is the supply capacity of supplier $i$.

3. Non-negativity:
$$
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
$$

Index Sets and Data Mapping

- $I$: Set of suppliers (distribution centers), from column "supplier_id" in supply_capacity.csv (table_id: file_1_view_0) and transportation_costs.csv (table_id: file_2_view_0).
- $J$: Set of customers, from column "customer_id" in customer_demand.csv (table_id: file_0_view_0) and as suffixes in transportation_costs.csv columns (table_id: file_2_view_0).

Parameter Mapping

- $d_j$: Demand for customer $j$ from "demand_units" in customer_demand.csv (table_id: file_0_view_0, column: demand_units, key: customer_id).
- $s_i$: Supply capacity for supplier $i$ from "supply_capacity_units" in supply_capacity.csv (table_id: file_1_view_0, column: supply_capacity_units, key: supplier_id).
- $c_{ij}$: Transportation cost per unit from supplier $i$ to customer $j$ from "transportation_cost_to_Ck" columns in transportation_costs.csv (table_id: file_2_view_0, row: supplier_id, column: transportation_cost_to_Ck, where $k$ matches customer_id).

Variable Domain

- $x_{ij} \geq 0$, continuous, for all $i \in I$, $j \in J$.

All index sets, parameters, and coefficients are mapped directly from the current source data as described above. No data is omitted or aggregated.