### Mathematical Model

Let:
- $I$ = set of sections, indexed by $i$ (from all SectionID in file_0_view_0)
- $J$ = set of products, indexed by $j$ (from all ProductName in file_1_view_0)

Parameters:
- $c_i$ = capacity of section $i$ (Capacity from file_0_view_0, indexed by SectionID)
- $v_j$ = value (price) of product $j$ (Value from file_1_view_0, indexed by ProductName)
- $w_j$ = space requirement of product $j$ (Weight from file_1_view_0, indexed by ProductName)

Decision variables:
- $x_{ij}$ = number of units of product $j$ to stock in section $i$ ($x_{ij} \in \mathbb{Z}_{\geq 0}$)

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

### Data Mapping

- $I$: All SectionID from file_0_view_0 (capacity.csv), column SectionID
- $J$: All ProductName from file_1_view_0 (products.csv), column ProductName
- $c_i$: file_0_view_0, Capacity, indexed by SectionID
- $v_j$: file_1_view_0, Value, indexed by ProductName
- $w_j$: file_1_view_0, Weight, indexed by ProductName
- $x_{ij}$: Decision variable for units of product $j$ in section $i$ (integer, $\geq 0$)