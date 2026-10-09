##### Decision Variables

Let $x_{ij} \geq 0$ be the continuous quantity shipped from distribution center $i$ to customer group $j$, for all $i \in I$, $j \in J$.

##### Sets

- $I$: set of distribution centers (from "supply_capacity.csv"): $I = \{\text{S1}, \text{S2}, \ldots, \text{S12}\}$
- $J$: set of customer groups (from "customer_demand.csv"): $J = \{\text{C1}, \text{C2}, \ldots, \text{C12}\}$

##### Parameters

- $d_j$: demand of customer group $j$ (from "customer_demand.csv", column "demand", table_id: file_0_view_0)
- $s_i$: supply capacity of distribution center $i$ (from "supply_capacity.csv", column "supply_capacity", table_id: file_1_view_0)
- $c_{ij}$: transportation cost per unit from distribution center $i$ to customer group $j$ (from "transportation_costs.csv", table_id: file_2_view_0, row $i$, column $j$)

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction:**  
   For each customer group $j \in J$,
   $$
   \sum_{i \in I} x_{ij} \geq d_j
   $$
2. **Supply capacity:**  
   For each distribution center $i \in I$,
   $$
   \sum_{j \in J} x_{ij} \leq s_i
   $$
3. **Non-negativity:**  
   $$
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   $$

##### Data Mapping

- $I$ (distribution centers): all "Unnamed: 0" values in "supply_capacity.csv" (table_id: file_1_view_0)
- $J$ (customer groups): all "customer" values in "customer_demand.csv" (table_id: file_0_view_0)
- $d_j$: "demand" column in "customer_demand.csv" (table_id: file_0_view_0), indexed by "customer"
- $s_i$: "supply_capacity" column in "supply_capacity.csv" (table_id: file_1_view_0), indexed by "Unnamed: 0"
- $c_{ij}$: value in "transportation_costs.csv" (table_id: file_2_view_0), row "Unnamed: 0" = $i$, column $j$ (column names "C1"..."C12")

All indices, parameters, and coefficients are to be taken exactly as listed in the returned tables, preserving source order and identifiers.