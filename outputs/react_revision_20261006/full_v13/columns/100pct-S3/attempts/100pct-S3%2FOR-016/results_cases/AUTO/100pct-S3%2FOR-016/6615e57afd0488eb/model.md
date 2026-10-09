#### Mathematical Model

Let:
- $I$ = set of distribution centers (suppliers), indexed by $i$, from all supplier_id in file_1_view_0.
- $J$ = set of customer groups, indexed by $j$, from all customer_id in file_0_view_0.
- $x_{ij} \geq 0$ = quantity shipped from distribution center $i$ to customer group $j$ (continuous).
- $d_j$ = demand of customer group $j$ (from file_0_view_0, demand_units).
- $s_i$ = supply capacity of distribution center $i$ (from file_1_view_0, supply_capacity_units).
- $c_{ij}$ = transportation cost per unit from $i$ to $j$ (from file_2_view_0, transportation_cost_to_C*).

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

- $I$ (distribution centers): All supplier_id in file_1_view_0 (supply_capacity.csv), source order.
- $J$ (customer groups): All customer_id in file_0_view_0 (customer_demand.csv), source order.
- $d_j$: file_0_view_0, column demand_units, indexed by customer_id.
- $s_i$: file_1_view_0, column supply_capacity_units, indexed by supplier_id.
- $c_{ij}$: file_2_view_0, row supplier_id, column transportation_cost_to_C*, where * matches customer_id.

All indices, coefficients, and constraints are mapped directly from the current CSV source data, preserving source order and identifiers. Variable domains and constraint senses are as specified in the user query.