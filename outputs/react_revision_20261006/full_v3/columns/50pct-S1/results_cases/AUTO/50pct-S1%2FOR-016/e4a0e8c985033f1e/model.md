##### Decision Variables

Let $x_{ij} \geq 0$ be the continuous quantity shipped from distribution center $i$ (supplier) to customer group $j$.

##### Sets

- $I$: set of distribution centers (suppliers), from the column "supplier_id" in "supply_capacity.csv" and "transportation_costs.csv".
- $J$: set of customer groups, from the column "customer_id" in "customer_demand.csv" and columns "transportation_cost_to_Ck" in "transportation_costs.csv".

##### Parameters

- $d_j$: demand of customer group $j$, from "demand_units" in "customer_demand.csv".
- $s_i$: supply capacity of distribution center $i$, from "supply_capacity_units" in "supply_capacity.csv".
- $c_{ij}$: transportation cost per unit from distribution center $i$ to customer group $j$, from "transportation_cost_to_Ck" in "transportation_costs.csv".

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction:** For each customer group $j \in J$,
$$
\sum_{i \in I} x_{ij} \geq d_j
$$

2. **Supply capacity:** For each distribution center $i \in I$,
$$
\sum_{j \in J} x_{ij} \leq s_i
$$

3. **Non-negativity:**
$$
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
$$

##### Data Mapping

- $I$ (distribution centers): all "supplier_id" in "supply_capacity.csv" and "transportation_costs.csv" (table_id: file_1_view_0, file_2_view_0)
- $J$ (customer groups): all "customer_id" in "customer_demand.csv" and columns "transportation_cost_to_Ck" in "transportation_costs.csv" (table_id: file_0_view_0, file_2_view_0)
- $d_j$: "demand_units" for customer $j$ in "customer_demand.csv" (table_id: file_0_view_0, column: demand_units)
- $s_i$: "supply_capacity_units" for supplier $i$ in "supply_capacity.csv" (table_id: file_1_view_0, column: supply_capacity_units)
- $c_{ij}$: "transportation_cost_to_Ck" for supplier $i$ and customer $j$ in "transportation_costs.csv" (table_id: file_2_view_0, row: supplier_id, column: transportation_cost_to_Ck)

All indices, parameters, and coefficients are to be taken exactly as listed in the source tables, preserving all identifiers and values.