#### Mathematical Model

Let:
- $I$ = set of sections (indexed by $i$), with SectionID from file_0_view_0.
- $J$ = set of products (indexed by $j$), with ProductName from file_1_view_0.

Parameters:
- $c_i$ = Capacity of section $i$ (from file_0_view_0, column Capacity, indexed by SectionID).
- $v_j$ = Value (price) of product $j$ (from file_1_view_0, column Value, indexed by ProductName).
- $w_j$ = Weight (space requirement) of product $j$ (from file_1_view_0, column Weight, indexed by ProductName).

Decision variables:
- $x_{ij}$ = number of units of product $j$ to stock in section $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$.

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

- $I$ (sections): All SectionID in file_0_view_0 (capacity.csv), column SectionID.
- $J$ (products): All ProductName in file_1_view_0 (products.csv), column ProductName.
- $c_i$: file_0_view_0, column Capacity, indexed by SectionID.
- $v_j$: file_1_view_0, column Value, indexed by ProductName.
- $w_j$: file_1_view_0, column Weight, indexed by ProductName.
- $x_{ij}$: Integer variable for each $(i,j)$ pair, as above.