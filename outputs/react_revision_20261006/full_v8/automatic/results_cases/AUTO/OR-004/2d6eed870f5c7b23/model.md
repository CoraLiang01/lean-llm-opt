##### Mathematical Model

Let $I$ be the set of distribution centers (indexed by $i$), and $J$ the set of customer groups (indexed by $j$):

- $I = \{\text{S1}, \text{S2}, \ldots, \text{S12}\}$, from column "Unnamed: 0" in supply_capacity.csv and transportation_costs.csv.
- $J = \{\text{C1}, \text{C2}, \ldots, \text{C12}\}$, from column "customer" in customer_demand.csv and columns in transportation_costs.csv.

Parameters:
- $d_j$: demand of customer group $j$ (from "demand" in customer_demand.csv, table_id file_0_view_0).
- $s_i$: supply capacity of distribution center $i$ (from "supply_capacity" in supply_capacity.csv, table_id file_1_view_0).
- $c_{ij}$: transportation cost per unit from distribution center $i$ to customer group $j$ (from transportation_costs.csv, table_id file_2_view_0, row "Unnamed: 0" = $i$, column $j$).

Decision variables:
- $x_{ij} \geq 0$: quantity shipped from distribution center $i$ to customer group $j$ (continuous).

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:
\[
\sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
\]
\[
\sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
\]
\[
x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
\]

##### Data Mapping

- $I$: All values in column "Unnamed: 0" of supply_capacity.csv (table_id file_1_view_0) and row labels of transportation_costs.csv (table_id file_2_view_0).
- $J$: All values in column "customer" of customer_demand.csv (table_id file_0_view_0) and columns of transportation_costs.csv (table_id file_2_view_0), excluding "Unnamed: 0".
- $d_j$: For each $j \in J$, value in "demand" column of customer_demand.csv (table_id file_0_view_0, row where "customer" = $j$).
- $s_i$: For each $i \in I$, value in "supply_capacity" column of supply_capacity.csv (table_id file_1_view_0, row where "Unnamed: 0" = $i$).
- $c_{ij}$: For each $i \in I$, $j \in J$, value in transportation_costs.csv (table_id file_2_view_0, row where "Unnamed: 0" = $i$, column $j$).

All index sets, parameters, and constraints are defined exactly as in the current source data. No data is omitted or aggregated. Variable domains and all constraints are as specified in the user query.