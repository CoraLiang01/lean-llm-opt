##### Mathematical Model

Let:
- $I$ = set of sections (indexed by $i$), with $i \in$ SectionID from file_0_view_0
- $J$ = set of products (indexed by $j$), with $j \in$ ProductName from file_1_view_0

Parameters:
- $c_i$ = Capacity of section $i$ (from file_0_view_0, column Capacity)
- $v_j$ = Value (price) of product $j$ (from file_1_view_0, column Value)
- $w_j$ = Weight (space requirement) of product $j$ (from file_1_view_0, column Weight)

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

##### Data Mapping

- $I$ (sections): file_0_view_0, column SectionID
- $J$ (products): file_1_view_0, column ProductName
- $c_i$: file_0_view_0, columns [SectionID, Capacity]
- $v_j$: file_1_view_0, columns [ProductName, Value]
- $w_j$: file_1_view_0, columns [ProductName, Weight]
- $x_{ij}$: decision variable for each $(i,j)$ pair

All indices, parameters, and constraints are defined directly from the supplied data.