##### Mathematical Model

Let:
- $I$ = set of production plants (indexed by $i$), from supplier_id in file_1_view_0 and file_2_view_0: $\{S1, S2, S3, S4\}$
- $J$ = set of retail outlets (indexed by $j$), from customer_id in file_0_view_0: $\{C1, C2, C3, C4\}$
- $x_{ij} \geq 0$ = quantity shipped from plant $i$ to outlet $j$ (continuous variable)
- $d_j$ = demand at outlet $j$ (from file_0_view_0)
- $s_i$ = supply capacity at plant $i$ (from file_1_view_0)
- $c_{ij}$ = transportation cost per unit from plant $i$ to outlet $j$ (from file_2_view_0)

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:
1. Demand satisfaction (each outlet receives at least its demand):
\[
\sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
\]
2. Supply capacity (each plant ships no more than its capacity):
\[
\sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
\]
3. Non-negativity:
\[
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
\]

##### Data Mapping

- $I$ (plants): supplier_id from file_1_view_0 and file_2_view_0: S1, S2, S3, S4
- $J$ (outlets): customer_id from file_0_view_0: C1, C2, C3, C4
- $d_j$: demand column in file_0_view_0, indexed by customer_id
- $s_i$: supply_capacity column in file_1_view_0, indexed by supplier_id
- $c_{ij}$: transportation_cost_to_Ck columns in file_2_view_0, with row supplier_id $i$ and column $j$ mapped as:
    - C1: transportation_cost_to_C1
    - C2: transportation_cost_to_C2
    - C3: transportation_cost_to_C3
    - C4: transportation_cost_to_C4
- $x_{ij}$: decision variable for each $(i,j) \in I \times J$ (continuous, $\geq 0$)