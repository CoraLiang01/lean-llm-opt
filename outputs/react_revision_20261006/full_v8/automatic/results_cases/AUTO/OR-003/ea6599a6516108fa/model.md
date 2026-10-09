#### Mathematical Model

Let:
- $I$ = set of suppliers, indexed by $i$ (from "supply_capacity.csv", column "Unnamed: 0")
- $J$ = set of customer groups, indexed by $j$ (from "customer_demand.csv", column "customer")
- $x_{ij} \geq 0$ = quantity shipped from supplier $i$ to customer group $j$ (continuous variable)
- $d_j$ = demand of customer group $j$ (from "customer_demand.csv", column "demand")
- $s_i$ = supply capacity of supplier $i$ (from "supply_capacity.csv", column "supply_capacity")
- $c_{ij}$ = transportation cost per unit from supplier $i$ to customer group $j$ (from "transportation_costs.csv", entry at row $i$, column $j$)

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

#### Data Mapping

- $I$ (suppliers): All values in "supply_capacity.csv", column "Unnamed: 0", table_id: file_1_view_0
- $J$ (customer groups): All values in "customer_demand.csv", column "customer", table_id: file_0_view_0
- $d_j$: "customer_demand.csv", column "demand", table_id: file_0_view_0, mapped by "customer"
- $s_i$: "supply_capacity.csv", column "supply_capacity", table_id: file_1_view_0, mapped by "Unnamed: 0"
- $c_{ij}$: "transportation_costs.csv", table_id: file_2_view_0, row "Unnamed: 0" = $i$, column $j$ (column names match "customer" in file_0_view_0)

All index sets, parameters, and coefficients are defined exactly as returned in the current Observation, preserving source order and identifiers.