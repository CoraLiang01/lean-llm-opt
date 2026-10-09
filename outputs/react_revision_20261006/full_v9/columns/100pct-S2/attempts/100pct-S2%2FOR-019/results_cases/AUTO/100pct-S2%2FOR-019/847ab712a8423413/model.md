#### Mathematical Model

Let:
- $I$ = set of distribution centers (indexed by $i$), from all supplier_id in supply_capacity.csv and transportation_costs.csv.
- $J$ = set of customer groups (indexed by $j$), from all customer_id in customer_demand.csv and transportation_costs.csv.
- $x_{ij} \geq 0$ = quantity shipped from distribution center $i$ to customer group $j$ (continuous).
- $d_j$ = demand of customer group $j$ (from customer_demand.csv).
- $s_i$ = supply capacity of distribution center $i$ (from supply_capacity.csv).
- $c_{ij}$ = transportation cost per unit from $i$ to $j$ (from transportation_costs.csv).

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

- $I$ (distribution centers): All supplier_id in supply_capacity.csv (file_1_view_0, column "supplier_id") and transportation_costs.csv (file_2_view_0, column "supplier_id").
- $J$ (customer groups): All customer_id in customer_demand.csv (file_0_view_0, column "customer_id") and as suffixes in transportation_costs.csv columns "transportation_cost_to_*".
- $d_j$: For each $j \in J$, from customer_demand.csv (file_0_view_0, columns "customer_id", "demand").
- $s_i$: For each $i \in I$, from supply_capacity.csv (file_1_view_0, columns "supplier_id", "supply_capacity").
- $c_{ij}$: For each $i \in I$, $j \in J$, from transportation_costs.csv (file_2_view_0, row "supplier_id" = $i$, column "transportation_cost_to_{j}").

Index sets, parameters, and all mappings are defined by the exact identifiers and values in the current CSV files, preserving source order and naming. No data is omitted or aggregated. Variable domains and all constraints are as specified above.