##### Mathematical Model

Let:
- $I$ = set of sections, indexed by $i$ (from all SectionID in file_0_view_0)
- $J$ = set of products, indexed by $j$ (from all ProductName in file_1_view_0)
- $x_{ij}$ = number of units of product $j$ to be placed in section $i$ (decision variable, integer, $\geq 0$)
- $v_j$ = value (revenue) per unit of product $j$ (from Value in file_1_view_0)
- $w_j$ = space requirement per unit of product $j$ (from Weight in file_1_view_0)
- $C_i$ = capacity of section $i$ (from Capacity in file_0_view_0)

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}
\]

**Subject to:**

Section capacity constraints:
\[
\sum_{j \in J} w_j \, x_{ij} \leq C_i \qquad \forall i \in I
\]

Nonnegativity and integrality:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
\]

---

##### Data Mapping

- $I$: All SectionID in file_0_view_0 (capacity.csv), column SectionID
- $J$: All ProductName in file_1_view_0 (products.csv), column ProductName
- $C_i$: file_0_view_0, column Capacity, keyed by SectionID
- $v_j$: file_1_view_0, column Value, keyed by ProductName
- $w_j$: file_1_view_0, column Weight, keyed by ProductName
- $x_{ij}$: Decision variable for each $(i,j) \in I \times J$