#### Mathematical Model

Let $I$ be the set of distribution centers (indexed by $i$), and $J$ the set of customer groups (indexed by $j$):

- $I = \{\text{S1}, \text{S2}, \ldots, \text{S18}\}$ (from supply_capacity.csv, column "Unnamed: 0")
- $J = \{\text{C1}, \text{C2}, \ldots, \text{C18}\}$ (from customer_demand.csv, column "customer")

Parameters:
- $d_j$: demand of customer group $j$ (from customer_demand.csv, column "demand")
- $s_i$: supply capacity of distribution center $i$ (from supply_capacity.csv, column "supply_capacity")
- $c_{ij}$: transportation cost per unit from distribution center $i$ to customer group $j$ (from transportation_costs.csv, row $i$, column $j$)

Decision variables:
- $x_{ij} \geq 0$: quantity shipped from distribution center $i$ to customer group $j$ (continuous)

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:
1. Demand satisfaction:
   \[
   \sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
   \]
2. Supply capacity:
   \[
   \sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
   \]
3. Non-negativity:
   \[
   x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   \]

#### Data Mapping

- $I$ (distribution centers): file_1_view_0, column "Unnamed: 0"
- $J$ (customer groups): file_0_view_0, column "customer"
- $d_j$: file_0_view_0, column "demand", row where "customer" = $j$
- $s_i$: file_1_view_0, column "supply_capacity", row where "Unnamed: 0" = $i$
- $c_{ij}$: file_2_view_0, row "Unnamed: 0" = $i$, column $j$ (column names "C1", ..., "C18")

- Decision variables $x_{ij}$: for all $i \in I$, $j \in J$, as above.

All index sets, parameters, and constraints are defined exactly as in the current CSV data, preserving all identifiers and source order.