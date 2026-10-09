##### Mathematical Model

Let $I$ be the set of distribution centers (suppliers) from column "supplier_id" in supply_capacity.csv and transportation_costs.csv, and $J$ the set of customer groups from column "customer_id" in customer_demand.csv and transportation_costs.csv.

Let $x_{ij} \geq 0$ be the continuous quantity shipped from supplier $i \in I$ to customer $j \in J$.

Parameters:
- $d_j$: demand of customer $j$ (from "demand" in customer_demand.csv)
- $s_i$: supply capacity of supplier $i$ (from "supply_capacity" in supply_capacity.csv)
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer $j$ (from "transportation_cost_to_Ck" in transportation_costs.csv, with $i$ from "supplier_id" and $j$ from "Ck$)

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

- $I$: All "supplier_id" in file_1_view_0 (supply_capacity.csv) and file_2_view_0 (transportation_costs.csv)
- $J$: All "customer_id" in file_0_view_0 (customer_demand.csv) and columns "transportation_cost_to_Ck" in file_2_view_0 (transportation_costs.csv)
- $d_j$: file_0_view_0, column "demand", row with "customer_id" = $j$
- $s_i$: file_1_view_0, column "supply_capacity", row with "supplier_id" = $i$
- $c_{ij}$: file_2_view_0, row with "supplier_id" = $i$, column "transportation_cost_to_Ck" where $j$ = $Ck$

Variables:
- $x_{ij}$: continuous, nonnegative, for all $i \in I$, $j \in J$