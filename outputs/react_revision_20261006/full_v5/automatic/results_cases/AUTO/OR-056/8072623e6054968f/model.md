#### Mathematical Optimization Model

Let:
- $I$ = set of display areas, indexed by $i$ (DisplayID from file_0_view_0)
- $J$ = set of boat types, indexed by $j$ (ProductName from file_1_view_0)

Parameters:
- $c_i$ = capacity of display area $i$ (Capacity from file_0_view_0)
- $v_j$ = value of boat type $j$ (Value from file_1_view_0)
- $w_j$ = size (weight) of boat type $j$ (Weight from file_1_view_0)

Decision variables:
- $x_{ij}$ = number of boats of type $j$ to place in display area $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

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

- $I$: DisplayID from file_0_view_0 (capacity.csv)
- $J$: ProductName from file_1_view_0 (products.csv)
- $c_i$: Capacity column in file_0_view_0, matched by DisplayID
- $v_j$: Value column in file_1_view_0, matched by ProductName
- $w_j$: Weight column in file_1_view_0, matched by ProductName
- $x_{ij}$: Number of boats of type $j$ in display area $i$ (decision variable, nonnegative integer)