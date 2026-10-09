##### Decision Variables

Let $x_{ij} \geq 0$ be the continuous quantity shipped from distribution center $i$ to customer group $j$, for all $i \in I$, $j \in J$.

##### Sets

- $I$: set of distribution centers (from "supply_capacity.csv"): $I = \{\text{S1}, \text{S2}, \ldots, \text{S18}\}$
- $J$: set of customer groups (from "customer_demand.csv"): $J = \{\text{C1}, \text{C2}, \ldots, \text{C18}\}$

##### Parameters

- $d_j$: demand of customer group $j$ (from "customer_demand.csv")
- $s_i$: supply capacity of distribution center $i$ (from "supply_capacity.csv")
- $c_{ij}$: transportation cost per unit from distribution center $i$ to customer group $j$ (from "transportation_costs.csv")

##### Objective

Minimize total transportation cost:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

##### Constraints

1. **Demand satisfaction:** Each customer group must receive at least its demand.
   $$
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   $$
2. **Supply capacity:** Each distribution center cannot ship more than its supply capacity.
   $$
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
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
- $c_{ij}$: entry in "transportation_costs.csv" (table_id: file_2_view_0), row "Unnamed: 0" = $i$, column $j$

All indices, parameters, and coefficients are to be taken exactly as listed in the respective CSV files.