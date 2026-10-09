#### Mathematical Model

Let:
- $S$ = set of sections (indexed by $i$), with SectionID from file_0_view_0
- $P$ = set of products (indexed by $j$), with ProductName from file_1_view_0

Parameters:
- $c_i$ = capacity of section $i$ (Capacity from file_0_view_0)
- $v_j$ = value (price) of product $j$ (Value from file_1_view_0)
- $w_j$ = space requirement of product $j$ (Weight from file_1_view_0)

Decision variables:
- $x_{ij}$ = number of units of product $j$ to stock in section $i$, $x_{ij} \in \mathbb{Z}_{\geq 0}$

Objective:
\[
\max \sum_{i \in S} \sum_{j \in P} v_j \, x_{ij}
\]

Subject to:
\[
\sum_{j \in P} w_j \, x_{ij} \leq c_i \qquad \forall i \in S
\]
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in S,\, j \in P
\]

---

#### Data Mapping

- $S$: SectionID from file_0_view_0 (capacity.csv)
- $c_i$: Capacity from file_0_view_0, indexed by SectionID
- $P$: ProductName from file_1_view_0 (products.csv)
- $v_j$: Value from file_1_view_0, indexed by ProductName
- $w_j$: Weight from file_1_view_0, indexed by ProductName
- $x_{ij}$: Number of units of product $j$ in section $i$ (decision variable, integer, $\geq 0$)