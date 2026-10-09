#### Abstract Symbolic Mathematical Model

Let:
- $I$ = set of distribution centers (suppliers), indexed by $i$ (from all supplier_id in supply_capacity.csv)
- $J$ = set of customer groups, indexed by $j$ (from all customer_id in customer_demand.csv)
- $d_j$ = demand of customer group $j$ (from demand_units in customer_demand.csv)
- $s_i$ = supply capacity of distribution center $i$ (from supply_capacity_units in supply_capacity.csv)
- $c_{ij}$ = transportation cost per unit from distribution center $i$ to customer group $j$ (from transportation_cost_to_C* columns in transportation_costs.csv)
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

#### Data Mapping

- $I$ (distribution centers): All supplier_id in table_id file_1_view_0 (supply_capacity.csv)
- $J$ (customer groups): All customer_id in table_id file_0_view_0 (customer_demand.csv)
- $d_j$: demand_units from file_0_view_0, column demand_units, for each $j$
- $s_i$: supply_capacity_units from file_1_view_0, column supply_capacity_units, for each $i$
- $c_{ij}$: transportation_cost_to_C* columns in file_2_view_0 (transportation_costs.csv), with row supplier_id $i$ and column mapped to customer_id $j$ as per relationships.column_id_mapping in the Observation
- $x_{ij}$: decision variable for each $(i,j) \in I \times J$ (continuous, nonnegative)

All index sets, parameters, and mappings are defined exactly as in the current CSV data and relationships. No data is omitted or aggregated.