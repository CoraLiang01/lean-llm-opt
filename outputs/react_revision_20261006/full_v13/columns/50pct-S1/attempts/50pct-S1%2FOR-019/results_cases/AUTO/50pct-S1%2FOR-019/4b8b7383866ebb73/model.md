#### Mathematical Model

Let:
- $I$ = set of distribution centers (suppliers) from file_1_view_0: $\{\text{supplier1}, \text{supplier2}, \text{supplier3}, \text{supplier4}, \text{supplier5}, \text{supplier6}, \text{supplier7}, \text{supplier8}\}$
- $J$ = set of customer groups from file_0_view_0: $\{\text{demand1}, \text{demand2}, \text{demand3}, \text{demand4}, \text{demand5}, \text{demand6}, \text{demand7}, \text{demand8}\}$
- $x_{ij} \geq 0$ = quantity shipped from supplier $i \in I$ to customer $j \in J$ (continuous)
- $d_j$ = demand of customer $j$ (from file_0_view_0)
- $s_i$ = supply capacity of supplier $i$ (from file_1_view_0)
- $c_{ij}$ = transportation cost per unit from supplier $i$ to customer $j$ (from file_2_view_0, column "transportation_cost_to_demandX" for each $j$)

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

- $I$ (suppliers): file_1_view_0, column "supplier_id", all rows
- $J$ (customers): file_0_view_0, column "customer_id", all rows
- $d_j$: file_0_view_0, column "demand", indexed by "customer_id"
- $s_i$: file_1_view_0, column "supply_capacity", indexed by "supplier_id"
- $c_{ij}$: file_2_view_0, row "supplier_id" = $i$, column "transportation_cost_to_demandX" where $X$ matches $j$ (e.g., "transportation_cost_to_demand3" for $j$ = "demand3")

All index sets, parameters, and constraints are defined exactly as in the current CSV data. No data is omitted or aggregated. Variable domains and all bounds are as specified in the user query and source data.