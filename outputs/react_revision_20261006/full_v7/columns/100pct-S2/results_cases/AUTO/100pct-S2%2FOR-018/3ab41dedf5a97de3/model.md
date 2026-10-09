##### Decision Variables

Let $x_{ij} \geq 0$ be the continuous quantity shipped from distribution center (supplier) $i \in I$ to customer group $j \in J$.

##### Sets

- $I = \{$S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12$\}$ (distribution centers)
- $J = \{$C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12$\}$ (customer groups)

##### Parameters

- $d_j$: demand of customer group $j \in J$ (from "customer_demand.csv")
- $s_i$: supply capacity of distribution center $i \in I$ (from "supply_capacity.csv")
- $c_{ij}$: transportation cost per unit from $i$ to $j$ (from "transportation_costs.csv")

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

- $I$ (distribution centers): supplier_id in "supply_capacity.csv" and "transportation_costs.csv"
- $J$ (customer groups): customer_id in "customer_demand.csv" and columns in "transportation_costs.csv" (suffix after "transportation_cost_to_")
- $d_j$: demand column in "customer_demand.csv", indexed by customer_id
- $s_i$: supply_capacity column in "supply_capacity.csv", indexed by supplier_id
- $c_{ij}$: "transportation_costs.csv", row supplier_id $i$, column "transportation_cost_to_$j$" where $j$ is customer_id

All indices, parameters, and coefficients are to be taken exactly as listed in the source files and relationships.