#### Mathematical Model

Let:
- $I$ = set of sections, indexed by $i$ (SectionID from file_0_view_0)
- $J$ = set of products, indexed by $j$ (ProductName from file_1_view_0)

Parameters:
- $c_i$ = capacity of section $i$ (Capacity from file_0_view_0)
- $v_j$ = value (price) of product $j$ (Value from file_1_view_0)
- $w_j$ = space requirement of product $j$ (Weight from file_1_view_0)

Decision variables:
- $x_{ij}$ = number of units of product $j$ to stock in section $i$, $x_{ij} \in \mathbb{Z}_{\geq 0}$

Objective:
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}
\]

Subject to:
\[
\sum_{j \in J} w_j \, x_{ij} \leq c_i \qquad \forall i \in I
\]
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
\]

---

#### Data Mapping

- $I$: SectionID from file_0_view_0 (capacity.csv)
- $J$: ProductName from file_1_view_0 (products.csv)
- $c_i$: Capacity column in file_0_view_0, keyed by SectionID
- $v_j$: Value column in file_1_view_0, keyed by ProductName
- $w_j$: Weight column in file_1_view_0, keyed by ProductName
- $x_{ij}$: Number of units of product $j$ to stock in section $i$ (decision variable, integer, nonnegative)