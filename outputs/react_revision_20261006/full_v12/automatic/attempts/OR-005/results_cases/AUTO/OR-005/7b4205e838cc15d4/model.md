#### Mathematical Model

Let $I$ be the set of distribution centers (suppliers) from file_1_view_0: $\{\text{supplier1},\ldots,\text{supplier8}\}$.

Let $J$ be the set of customer groups from file_0_view_0: $\{\text{demand1},\ldots,\text{demand8}\}$.

Let $x_{ij} \geq 0$ be the quantity shipped from supplier $i \in I$ to customer $j \in J$ (continuous).

Parameters:
- $d_j$: demand of customer $j$ (from file_0_view_0, column "demand")
- $s_i$: supply capacity of supplier $i$ (from file_1_view_0, column "supply_capacity")
- $c_{ij}$: transportation cost per unit from supplier $i$ to customer $j$ (from file_2_view_0, row "Unnamed: 0" = supply*, column = demand*)

Objective:
\[
\min \sum_{i \in I} \sum_{j \in J} c_{ij} x_{ij}
\]

Subject to:
\[
\sum_{i \in I} x_{ij} \geq d_j \quad \forall j \in J
\]
\[
\sum_{j \in J} x_{ij} \leq s_i \quad \forall i \in I
\]
\[
x_{ij} \geq 0 \quad \forall i \in I,\, j \in J
\]

#### Data Mapping

- $I$: supplier IDs from file_1_view_0, column "Supplier"
- $J$: customer IDs from file_0_view_0, column "Customers"
- $d_j$: file_0_view_0, column "demand", row where "Customers" = $j$
- $s_i$: file_1_view_0, column "supply_capacity", row where "Supplier" = $i$
- $c_{ij}$: file_2_view_0, row "Unnamed: 0" = supply*, column = demand* (matrix: suppliers × customers, aligned by IDs as above)
- $x_{ij}$: decision variable for each $(i,j) \in I \times J$