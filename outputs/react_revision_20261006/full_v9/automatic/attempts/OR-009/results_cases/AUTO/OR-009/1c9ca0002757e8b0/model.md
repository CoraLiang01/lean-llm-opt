#### Mathematical Model

Let:
- $I$ = set of production plants (indexed by $i$), from file_1_view_0: $\{S1, S2, S3, S4\}$
- $J$ = set of retail outlets (indexed by $j$), from file_0_view_0: $\{C1, C2, C3, C4\}$
- $x_{ij} \geq 0$ = quantity shipped from plant $i$ to outlet $j$ (continuous variable)
- $d_j$ = demand of outlet $j$ (from file_0_view_0)
- $s_i$ = supply capacity of plant $i$ (from file_1_view_0)
- $c_{ij}$ = transportation cost per unit from plant $i$ to outlet $j$ (from file_2_view_0)

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

- $I$ (plants): all "Unnamed: 0" in file_1_view_0 (S1, S2, S3, S4)
- $J$ (retail outlets): all "customer" in file_0_view_0 (C1, C2, C3, C4)
- $d_j$: "demand" column in file_0_view_0, indexed by "customer"
- $s_i$: "supply_capacity" column in file_1_view_0, indexed by "Unnamed: 0"
- $c_{ij}$: file_2_view_0, with rows indexed by "Unnamed: 0" (plants), columns by retail outlet IDs (C1, C2, C3, C4)
- $x_{ij}$: decision variable for each $(i,j) \in I \times J$

All index sets, parameters, and coefficients are mapped directly from the current CSV files as described above. No data is omitted or aggregated. Variable domains and all constraints are as specified in the user query.