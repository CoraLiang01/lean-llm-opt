##### Decision Variables

$x_{ij} \geq 0$: quantity shipped from distribution center $i \in I$ to customer group $j \in J$ (continuous).

##### Objective Function

$\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}$

##### Constraints

1. Demand satisfaction: $\sum_{i \in I} x_{ij} \geq d_j,\quad \forall j \in J$
2. Supply capacity: $\sum_{j \in J} x_{ij} \leq s_i,\quad \forall i \in I$
3. Non-negativity: $x_{ij} \geq 0$ (continuous)

Where:

- $I = \{$S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12, S13, S14, S15, S16, S17, S18$\}$
- $J = \{$C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12, C13, C14, C15, C16, C17, C18$\}$

Parameters:

- $d_j$: demand of customer $j$ (from customer_demand.csv)
- $s_i$: supply capacity of distribution center $i$ (from supply_capacity.csv)
- $c_{ij}$: transportation cost per unit from distribution center $i$ to customer $j$ (from transportation_costs.csv)

##### Data Mapping

- $d_j$: "customer_demand.csv", column "customer" (identifier), column "demand" (value), source order.
- $s_i$: "supply_capacity.csv", column "Unnamed: 0" (identifier), column "supply_capacity" (value), source order.
- $c_{ij}$: "transportation_costs.csv", row "Unnamed: 0" (distribution center $i$), columns "C1"–"C18" (customer $j$), source order.