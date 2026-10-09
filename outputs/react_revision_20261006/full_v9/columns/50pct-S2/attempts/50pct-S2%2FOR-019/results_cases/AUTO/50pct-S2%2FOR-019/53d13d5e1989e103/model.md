##### Abstract Symbolic Mathematical Model

Let:
- $I$ = set of distribution centers (suppliers), indexed by $i$ (from all supplier_id in supply_capacity.csv and transportation_costs.csv)
- $J$ = set of customer groups, indexed by $j$ (from all customer_id in customer_demand.csv and transportation_costs.csv)
- $d_j$ = demand of customer group $j$ (from customer_demand.csv)
- $s_i$ = supply capacity of distribution center $i$ (from supply_capacity.csv)
- $c_{ij}$ = transportation cost per unit from distribution center $i$ to customer group $j$ (from transportation_costs.csv)
- $x_{ij} \geq 0$ = quantity shipped from distribution center $i$ to customer group $j$ (continuous)

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

- $I$ (distribution centers): All unique supplier_id in supply_capacity.csv (file_1_view_0, column "supplier_id") and transportation_costs.csv (file_2_view_0, column "supplier_id")
- $J$ (customer groups): All unique customer_id in customer_demand.csv (file_0_view_0, column "customer_id") and as suffixes in transportation_costs.csv columns "transportation_cost_to_*"
- $d_j$: For each $j \in J$, demand from customer_demand.csv (file_0_view_0, columns "customer_id", "demand")
- $s_i$: For each $i \in I$, supply_capacity from supply_capacity.csv (file_1_view_0, columns "supplier_id", "supply_capacity")
- $c_{ij}$: For each $i \in I$, $j \in J$, transportation_costs.csv (file_2_view_0, row "supplier_id" = $i$, column "transportation_cost_to_{j}")
- $x_{ij}$: Decision variable for each $i \in I$, $j \in J$

Index sets, parameters, and cost matrix are defined strictly by the current CSV data, preserving all identifiers and source order. No data is omitted or aggregated. All constraints and variable domains are as specified in the user query.