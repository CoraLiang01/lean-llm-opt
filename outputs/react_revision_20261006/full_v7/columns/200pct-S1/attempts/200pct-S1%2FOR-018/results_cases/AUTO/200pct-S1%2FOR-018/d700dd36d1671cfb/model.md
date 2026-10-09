##### Decision Variables

Let $x_{ij} \geq 0$ be the continuous quantity shipped from distribution center (supplier) $i \in I$ to customer group $j \in J$.

##### Sets

- $I$: set of distribution centers (suppliers), from "supply_capacity.csv":
  $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}, \text{S5}, \text{S6}, \text{S7}, \text{S8}, \text{S9}, \text{S10}, \text{S11}, \text{S12}\}$
- $J$: set of customer groups, from "customer_demand.csv":
  $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}, \text{C5}, \text{C6}, \text{C7}, \text{C8}, \text{C9}, \text{C10}, \text{C11}, \text{C12}\}$

##### Parameters

- $d_j$: demand of customer group $j \in J$, from "customer_demand.csv"
- $s_i$: supply capacity of distribution center $i \in I$, from "supply_capacity.csv"
- $c_{ij}$: transportation cost per unit from $i$ to $j$, from "transportation_costs.csv"

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

- $I$ (distribution centers): supplier_id column in "supply_capacity.csv" and "transportation_costs.csv"
- $J$ (customer groups): customer_id column in "customer_demand.csv" and suffix of transportation_cost_to_C* columns in "transportation_costs.csv"
- $d_j$: demand column in "customer_demand.csv", indexed by customer_id
- $s_i$: supply_capacity column in "supply_capacity.csv", indexed by supplier_id
- $c_{ij}$: transportation_cost_to_C* columns in "transportation_costs.csv", indexed by supplier_id (rows) and customer_id (columns)

All indices, parameters, and coefficients are to be taken exactly as listed in the respective CSV files.