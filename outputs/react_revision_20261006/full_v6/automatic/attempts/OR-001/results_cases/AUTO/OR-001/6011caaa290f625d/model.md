##### Decision Variables

Let $x_{ij} \geq 0$ be the continuous quantity shipped from distribution center $i$ to customer group $j$.

##### Sets

- $I$: set of distribution centers (from supply_capacity.csv): $I = \{\text{S1}, \text{S2}, \ldots, \text{S18}\}$
- $J$: set of customer groups (from customer_demand.csv): $J = \{\text{C1}, \text{C2}, \ldots, \text{C18}\}$

##### Parameters

- $d_j$: demand of customer group $j$ (from customer_demand.csv)
- $s_i$: supply capacity of distribution center $i$ (from supply_capacity.csv)
- $c_{ij}$: transportation cost per unit from distribution center $i$ to customer group $j$ (from transportation_costs.csv)

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
3. **Non-negativity:** For all $i \in I$, $j \in J$,
   $$
   x_{ij} \geq 0
   $$

##### Data Mapping

- $I$ (distribution centers): all values in column "Unnamed: 0" of supply_capacity.csv and transportation_costs.csv, in source order.
- $J$ (customer groups): all values in column "customer" of customer_demand.csv and all columns (except "Unnamed: 0") of transportation_costs.csv, in source order.
- $d_j$: parameter from "demand" column of customer_demand.csv, indexed by "customer".
- $s_i$: parameter from "supply_capacity" column of supply_capacity.csv, indexed by "Unnamed: 0".
- $c_{ij}$: parameter from transportation_costs.csv, with row index "Unnamed: 0" (distribution center) and column index $j$ (customer group).

All indices, parameters, and coefficients are to be taken exactly as listed in the source files, preserving their order and identifiers.