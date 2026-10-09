#### Mathematical Model

Let $I = \{\text{S1}, \text{S2}, \text{S3}, \text{S4}\}$ (production plants) and $J = \{\text{C1}, \text{C2}, \text{C3}, \text{C4}\}$ (retail outlets).

Decision variables:
- $x_{ij} \geq 0$: quantity shipped from plant $i \in I$ to outlet $j \in J$ (continuous).

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]
where $c_{ij}$ is the transportation cost per unit from plant $i$ to outlet $j$.

Subject to:
1. Demand satisfaction (each outlet receives at least its demand):
   \[
   \sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
   \]
   where $d_j$ is the demand of outlet $j$.

2. Supply capacity (each plant ships no more than its capacity):
   \[
   \sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
   \]
   where $s_i$ is the supply capacity of plant $i$.

3. Non-negativity:
   \[
   x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
   \]

#### Data Mapping

- $I$ (plants): supplier_id from supply_capacity.csv (file_1_view_0)
- $J$ (outlets): customer_id from customer_demand.csv (file_0_view_0)
- $d_j$: demand column in customer_demand.csv (file_0_view_0), indexed by customer_id
- $s_i$: supply_capacity column in supply_capacity.csv (file_1_view_0), indexed by supplier_id
- $c_{ij}$: transportation_cost_to_Ck columns in transportation_costs.csv (file_2_view_0), with supplier_id as rows and C1–C4 as columns

All indices, parameters, and constraints are mapped directly to the current source data as described above.