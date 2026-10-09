#### Mathematical Model

Let:
- $I$ = set of production plants = $\{\text{S1}, \text{S2}, \text{S3}, \text{S4}\}$ (from file_1_view_0.supplier_id)
- $J$ = set of retail outlets = $\{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}$ (from file_0_view_0.customer_id)
- $d_j$ = demand at outlet $j$ (from file_0_view_0.demand)
- $s_i$ = supply capacity at plant $i$ (from file_1_view_0.supply_capacity)
- $c_{ij}$ = transportation cost per unit from plant $i$ to outlet $j$ (from file_2_view_0, columns transportation_cost_to_C1, ..., transportation_cost_to_C4, indexed by supplier_id)

Decision variables:
- $x_{ij} \geq 0$: quantity shipped from plant $i$ to outlet $j$ (continuous)

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:
1. Demand satisfaction (for each outlet $j$):
   \[
   \sum_{i \in I} x_{ij} \geq d_j \qquad \forall j \in J
   \]
2. Plant capacity (for each plant $i$):
   \[
   \sum_{j \in J} x_{ij} \leq s_i \qquad \forall i \in I
   \]
3. Non-negativity:
   \[
   x_{ij} \geq 0 \qquad \forall i \in I,\, j \in J
   \]

#### Data Mapping

- $I$ (plants): file_1_view_0.supplier_id
- $J$ (outlets): file_0_view_0.customer_id
- $d_j$: file_0_view_0.demand, indexed by customer_id
- $s_i$: file_1_view_0.supply_capacity, indexed by supplier_id
- $c_{ij}$: file_2_view_0, columns transportation_cost_to_C1, transportation_cost_to_C2, transportation_cost_to_C3, transportation_cost_to_C4, indexed by supplier_id (rows) and customer_id (columns)
- $x_{ij}$: decision variable for each $(i,j) \in I \times J$ (continuous, $\geq 0$)