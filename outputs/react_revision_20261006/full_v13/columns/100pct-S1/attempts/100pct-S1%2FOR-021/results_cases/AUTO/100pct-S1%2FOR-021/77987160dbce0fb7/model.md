##### Mathematical Model

Let $I$ be the set of production plants (indexed by $i$), and $J$ the set of retail outlets (indexed by $j$):

- $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}\}$
- $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}$

Let:
- $x_{ij} \geq 0$: quantity of beverages shipped from plant $i \in I$ to outlet $j \in J$ (continuous variable)
- $d_j$: demand of outlet $j$ (from Data Mapping)
- $s_i$: supply capacity of plant $i$ (from Data Mapping)
- $c_{ij}$: transportation cost per unit from plant $i$ to outlet $j$ (from Data Mapping)

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

- $I$ (plants): all supplier_id in file_1_view_0 (supply_capacity.csv): S1, S2, S3, S4
- $J$ (outlets): all customer_id in file_0_view_0 (customer_demand.csv): C1, C2, C3, C4
- $d_j$: demand for outlet $j$ from column demand in file_0_view_0, indexed by customer_id
- $s_i$: supply_capacity for plant $i$ from column supply_capacity in file_1_view_0, indexed by supplier_id
- $c_{ij}$: transportation_cost_to_Ck for plant $i$ and outlet $j$ from file_2_view_0 (transportation_costs.csv), where $i$ = supplier_id, $j$ = Ck (column transportation_cost_to_Ck)

Index sets, parameters, and all coefficients are mapped exactly as above from the current source data.