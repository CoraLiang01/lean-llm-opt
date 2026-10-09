#### Mathematical Model

Let $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}\}$ (production plants) and $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}$ (retail outlets).

Decision variables:
- $x_{ij} \geq 0$: quantity shipped from plant $i \in I$ to outlet $j \in J$ (continuous).

Parameters:
- $d_j$: demand at outlet $j$ (from file_0_view_0, column "demand").
- $s_i$: supply capacity at plant $i$ (from file_1_view_0, column "supply_capacity").
- $c_{ij}$: transportation cost per unit from plant $i$ to outlet $j$ (from file_2_view_0, columns "transportation_cost_to_C1", ..., "transportation_cost_to_C4").

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:
- Demand satisfaction:
  \[
  \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
  \]
- Supply capacity:
  \[
  \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
  \]
- Nonnegativity:
  \[
  x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
  \]

#### Data Mapping

- $I$ (plants): all "supplier_id" in file_1_view_0 and file_2_view_0, in source order.
- $J$ (outlets): all "customer_id" in file_0_view_0 and columns "transportation_cost_to_C1", ..., "transportation_cost_to_C4" in file_2_view_0, in source order.
- $d_j$: file_0_view_0, column "demand", indexed by "customer_id".
- $s_i$: file_1_view_0, column "supply_capacity", indexed by "supplier_id".
- $c_{ij}$: file_2_view_0, row "supplier_id" $i$, column "transportation_cost_to_{j}" for $j$ in $J$.

All indices, parameters, and constraints are mapped directly to the current source data as described.