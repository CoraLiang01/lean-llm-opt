#### Mathematical Model

Let $I$ be the set of distribution centers (suppliers) and $J$ the set of customer groups (demands):

- $I = \{\text{supplier1}, \text{supplier2}, \text{supplier3}, \text{supplier4}, \text{supplier5}, \text{supplier6}, \text{supplier7}, \text{supplier8}\}$
- $J = \{\text{demand1}, \text{demand2}, \text{demand3}, \text{demand4}, \text{demand5}, \text{demand6}, \text{demand7}, \text{demand8}\}$

Let $x_{ij} \geq 0$ be the continuous quantity shipped from supplier $i \in I$ to customer group $j \in J$.

Parameters:
- $d_j$: demand of customer group $j$ (from file_0_view_0, column "demand")
- $s_i$: supply capacity of supplier $i$ (from file_1_view_0, column "supply_capacity")
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer group $j$ (from file_2_view_0, columns "transportation_cost_to_demand1" ... "transportation_cost_to_demand8")

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

- $I$ (suppliers): file_1_view_0, column "supplier_id"
- $J$ (customer groups): file_0_view_0, column "customer_id"
- $d_j$: file_0_view_0, column "demand", indexed by "customer_id"
- $s_i$: file_1_view_0, column "supply_capacity", indexed by "supplier_id"
- $c_{ij}$: file_2_view_0, columns "transportation_cost_to_demand1" ... "transportation_cost_to_demand8", rows indexed by "supplier_id" (row axis), columns mapped to "customer_id" (column axis) as per relationships in the Observation

All indices, parameters, and mappings are as returned in the current Observation. No data is omitted or aggregated. Variable domains and all constraints are as specified in the user query.