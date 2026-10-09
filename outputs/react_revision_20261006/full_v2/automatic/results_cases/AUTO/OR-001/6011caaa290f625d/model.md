##### Decision Variables

Let $x_{ij} \geq 0$ be the continuous quantity shipped from distribution center $i$ to customer group $j$.

##### Sets

- $I$: set of distribution centers, indexed by $i$, from the "Unnamed: 0" column of "supply_capacity.csv" and "transportation_costs.csv" rows.
- $J$: set of customer groups, indexed by $j$, from the "customer" column of "customer_demand.csv" and "transportation_costs.csv" columns.

##### Parameters

- $d_j$: demand of customer group $j$, from "demand" column of "customer_demand.csv".
- $s_i$: supply capacity of distribution center $i$, from "supply_capacity" column of "supply_capacity.csv".
- $c_{ij}$: transportation cost per unit from distribution center $i$ to customer group $j$, from "transportation_costs.csv" (row $i$, column $j$).

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

- $I$ = all values in "Unnamed: 0" column of "supply_capacity.csv" and "transportation_costs.csv" rows.
- $J$ = all values in "customer" column of "customer_demand.csv" and "transportation_costs.csv" columns (excluding "Unnamed: 0").
- $d_j$ = "demand" column in "customer_demand.csv", indexed by "customer".
- $s_i$ = "supply_capacity" column in "supply_capacity.csv", indexed by "Unnamed: 0".
- $c_{ij}$ = value in "transportation_costs.csv" at row with "Unnamed: 0" = $i$, column $j$.

All indices, parameters, and coefficients are to be taken exactly as listed in the returned tables.