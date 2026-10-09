##### Decision Variables

Let $x_{ij} \geq 0$ be the continuous quantity shipped from distribution center $i$ to customer group $j$.

##### Sets

- $I$: set of distribution centers (from "supply_capacity.csv", column "Unnamed: 0")
- $J$: set of customer groups (from "customer_demand.csv", column "customer")

##### Parameters

- $d_j$: demand of customer group $j$ (from "customer_demand.csv", column "demand")
- $s_i$: supply capacity of distribution center $i$ (from "supply_capacity.csv", column "supply_capacity")
- $c_{ij}$: transportation cost per unit from distribution center $i$ to customer group $j$ (from "transportation_costs.csv", row $i$, column $j$)

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. Demand satisfaction for each customer group:
$$
\sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
$$

2. Supply capacity for each distribution center:
$$
\sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
$$

3. Non-negativity:
$$
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
$$

##### Data Mapping

- $I$ = all values in "supply_capacity.csv", column "Unnamed: 0", table_id: file_1_view_0
- $J$ = all values in "customer_demand.csv", column "customer", table_id: file_0_view_0
- $d_j$ = "demand" in "customer_demand.csv", table_id: file_0_view_0, for each $j \in J$
- $s_i$ = "supply_capacity" in "supply_capacity.csv", table_id: file_1_view_0, for each $i \in I$
- $c_{ij}$ = value in "transportation_costs.csv", table_id: file_2_view_0, row "Unnamed: 0" = $i$, column $j$ (where $i \in I$, $j \in J$)