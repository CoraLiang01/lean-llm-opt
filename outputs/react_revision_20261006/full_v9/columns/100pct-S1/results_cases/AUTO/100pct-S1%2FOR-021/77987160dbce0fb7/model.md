##### Mathematical Model

Let:
- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}\}$ (production plants, from supply_capacity.csv)
- $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}$ (retail outlets, from customer_demand.csv)
- $d_j$ = demand of outlet $j$ (from customer_demand.csv)
- $s_i$ = supply capacity of plant $i$ (from supply_capacity.csv)
- $c_{ij}$ = transportation cost per unit from plant $i$ to outlet $j$ (from transportation_costs.csv)
- $x_{ij} \geq 0$ = quantity shipped from plant $i$ to outlet $j$ (continuous)

Objective:
$$
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
$$

Subject to:
- Demand satisfaction:
  $$
  \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
  $$
- Supply capacity:
  $$
  \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
  $$
- Nonnegativity:
  $$
  x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
  $$

##### Data Mapping

- $I$ (plants): supplier_id in supply_capacity.csv (file_1_view_0)
- $J$ (outlets): customer_id in customer_demand.csv (file_0_view_0)
- $d_j$: demand column in customer_demand.csv (file_0_view_0), indexed by customer_id
- $s_i$: supply_capacity column in supply_capacity.csv (file_1_view_0), indexed by supplier_id
- $c_{ij}$: transportation_cost_to_Ck columns in transportation_costs.csv (file_2_view_0), with supplier_id as row index and Ck as column index (see relationships in Observation)
- $x_{ij}$: decision variable for each $(i,j) \in I \times J$ (continuous, nonnegative)

All index sets, parameters, and constraints are mapped directly to the current CSV data as described above.