##### Decision Variables

Let $x_{ij} \geq 0$ be the continuous quantity shipped from distribution center (supplier) $i \in I$ to customer group $j \in J$.

##### Sets

- $I$: set of distribution centers (suppliers), from the "supplier_id" column of "supply_capacity.csv" and "transportation_costs.csv".
- $J$: set of customer groups, from the "customer_id" column of "customer_demand.csv" and the cost matrix columns.

##### Parameters

- $d_j$: demand of customer group $j \in J$, from "customer_demand.csv".
- $s_i$: supply capacity of distribution center $i \in I$, from "supply_capacity.csv".
- $c_{ij}$: transportation cost per unit from distribution center $i$ to customer group $j$, from "transportation_costs.csv".

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

- $I$ (distribution centers): all "supplier_id" in "supply_capacity.csv" and "transportation_costs.csv" (S1, S2, ..., S12).
- $J$ (customer groups): all "customer_id" in "customer_demand.csv" and cost matrix columns (C1, C2, ..., C12).
- $d_j$: "demand" column in "customer_demand.csv", indexed by "customer_id" ($j$).
- $s_i$: "supply_capacity" column in "supply_capacity.csv", indexed by "supplier_id" ($i$).
- $c_{ij}$: "transportation_cost_to_Ck" columns in "transportation_costs.csv", with $i$ = "supplier_id", $j$ = $Ck$.

##### Matrix Structure

- Cost matrix: "transportation_costs.csv" (table_id: file_2_view_0), rows indexed by "supplier_id" ($i$), columns by "transportation_cost_to_Ck" ($j$).
- Demand vector: "customer_demand.csv" (table_id: file_0_view_0), "customer_id" ($j$), "demand".
- Supply vector: "supply_capacity.csv" (table_id: file_1_view_0), "supplier_id" ($i$), "supply_capacity".

All indices, parameters, and coefficients are to be taken exactly as listed in the source files and columns.