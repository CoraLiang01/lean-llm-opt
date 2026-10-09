#### Mathematical Model

Let $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}\}$ (production plants) and $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}$ (retail outlets).

Decision variables:
- $x_{ij} \geq 0$: quantity shipped from plant $i \in I$ to outlet $j \in J$ (continuous).

Parameters:
- $d_j$: demand at outlet $j \in J$.
- $s_i$: supply capacity at plant $i \in I$.
- $c_{ij}$: transportation cost per unit from plant $i$ to outlet $j$.

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:
1. Demand satisfaction:
   \[
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   \]
2. Plant capacity:
   \[
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   \]
3. Non-negativity:
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]

#### Data Mapping

- $I$ (plants): All supplier_id in file_1_view_0 (supply_capacity.csv): S1, S2, S3, S4
- $J$ (outlets): All customer_id in file_0_view_0 (customer_demand.csv): C1, C2, C3, C4
- $d_j$: demand column in file_0_view_0, indexed by customer_id
- $s_i$: supply_capacity column in file_1_view_0, indexed by supplier_id
- $c_{ij}$: transportation_cost_to_C* columns in file_2_view_0, indexed by supplier_id (rows) and customer_id (columns) as mapped in Observation

All indices, parameters, and coefficients are mapped directly from the current CSV files as described above.