##### Decision Variables

Let $x_{ij} \geq 0$ be the continuous quantity shipped from distribution center (supplier) $i \in I$ to customer group $j \in J$.

##### Sets

- $I$: set of distribution centers (suppliers), from the "supplier_id" column of "supply_capacity.csv" and "transportation_costs.csv".
- $J$: set of customer groups, from the "customer_id" column of "customer_demand.csv" and the columns of "transportation_costs.csv" (excluding "supplier_id").

##### Parameters

- $d_j$: demand (units) for customer group $j \in J$, from "customer_demand.csv".
- $s_i$: supply capacity (units) for distribution center $i \in I$, from "supply_capacity.csv".
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

- $I$ (distribution centers): all "supplier_id" in "supply_capacity.csv" and "transportation_costs.csv" (S1, S2, ..., S18).
- $J$ (customer groups): all "customer_id" in "customer_demand.csv" and columns "transportation_cost_to_C1", ..., "transportation_cost_to_C18" in "transportation_costs.csv" (C1, C2, ..., C18).
- $d_j$: "demand_units" from "customer_demand.csv", indexed by "customer_id" $j$.
- $s_i$: "supply_capacity_units" from "supply_capacity.csv", indexed by "supplier_id" $i$.
- $c_{ij}$: "transportation_cost_to_Ck" from "transportation_costs.csv", where $i$ is "supplier_id" and $j$ is $Ck$.

##### Matrix Structure

- The cost matrix $[c_{ij}]$ is given by "transportation_costs.csv", with rows indexed by "supplier_id" and columns by "transportation_cost_to_Ck" (mapping to $j = Ck$).
- The variable matrix $[x_{ij}]$ has the same structure: $i \in I$ (rows), $j \in J$ (columns).

##### Summary

Find $x_{ij} \geq 0$ for all $i \in I$, $j \in J$ to minimize total cost, such that all customer demands are met and no distribution center exceeds its supply capacity, using the exact data and identifiers from the provided CSV files.