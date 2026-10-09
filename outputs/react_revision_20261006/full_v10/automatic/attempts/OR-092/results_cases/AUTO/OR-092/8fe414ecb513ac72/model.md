##### Mathematical Model

Let:
- $I$ = set of sources (from expanded_sources.csv): $I = \{\text{S1}, \text{S2}, ..., \text{S10}\}$
- $J$ = set of destinations (from expanded_destinations.csv): $J = \{\text{D1}, \text{D2}, ..., \text{D20}\}$
- $c_{ij}$ = unit transportation cost from source $i$ to destination $j$ (from expanded_cost_matrix.csv)
- $s_i$ = supply at source $i$ (from expanded_sources.csv)
- $d_j$ = demand at destination $j$ (from expanded_destinations.csv)
- $t_{ij}$ = number of trucks dispatched from $i$ to $j$ (integer, $\geq 0$)
- $x_{ij}$ = quantity shipped from $i$ to $j$ (continuous, $0 \leq x_{ij} \leq 10 t_{ij}$)

Each truck can carry up to 10 units; partial loading is allowed, but $t_{ij}$ must be integer.

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:
\[
\sum_{i \in I} x_{ij} = d_j \quad \forall j \in J
\]
\[
\sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
\]
\[
0 \leq x_{ij} \leq 10 t_{ij} \quad \forall i \in I,\, j \in J
\]
\[
t_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J
\]
\[
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
\]

##### Data Mapping

- $I$ (sources): all source_id in expanded_sources.csv (file_2_view_0, column "source_id")
- $J$ (destinations): all destination_id in expanded_destinations.csv (file_1_view_0, column "destination_id")
- $c_{ij}$: expanded_cost_matrix.csv (file_0_view_0), row "source_id" $i$, column $j$
- $s_i$: expanded_sources.csv (file_2_view_0), column "supply_units" for source_id $i$
- $d_j$: expanded_destinations.csv (file_1_view_0), column "demand_units" for destination_id $j$
- $x_{ij}$: continuous variable, quantity shipped from $i$ to $j$
- $t_{ij}$: integer variable, number of trucks dispatched from $i$ to $j$