#### Mathematical Model

Let:
- $I$ = set of display areas, indexed by $i$ (DisplayID from file_0_view_0)
- $J$ = set of boat types, indexed by $j$ (ProductName from file_1_view_0)
- $c_i$ = capacity of display area $i$ (Capacity from file_0_view_0)
- $v_j$ = value of boat type $j$ (Value from file_1_view_0)
- $w_j$ = size of boat type $j$ (Weight from file_1_view_0)
- $x_{ij}$ = number of units of boat type $j$ placed in display area $i$ (decision variable, integer, $\geq 0$)

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
- $c_i$: Capacity from file_0_view_0, column "Capacity", key "DisplayID"
- $v_j$: Value from file_1_view_0, column "Value", key "ProductName"
- $w_j$: Weight from file_1_view_0, column "Weight", key "ProductName"
- $x_{ij}$: Number of units of boat type $j$ in display area $i$ (decision variable, integer, $\geq 0$)