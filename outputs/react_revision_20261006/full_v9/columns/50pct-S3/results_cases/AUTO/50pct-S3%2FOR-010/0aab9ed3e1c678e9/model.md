#### Mathematical Model

Let:
- $I$ = set of sections, indexed by $i$ (SectionID from file_0_view_0)
- $J$ = set of products, indexed by $j$ (ProductName from file_1_view_0)
- $x_{ij}$ = number of units of product $j$ to stock in section $i$ (decision variable, integer, $\geq 0$)
- $v_j$ = value (revenue) per unit of product $j$ (Value from file_1_view_0)
- $w_j$ = space requirement per unit of product $j$ (Weight from file_1_view_0)
- $C_i$ = display space capacity of section $i$ (Capacity from file_0_view_0)

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}
\]

**Subject to:**

- Section capacity constraints:
\[
\sum_{j \in J} w_j \, x_{ij} \leq C_i \qquad \forall i \in I
\]

- Nonnegativity and integrality:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
\]

---

#### Data Mapping

- $I$: SectionID from file_0_view_0 (capacity.csv)
- $J$: ProductName from file_1_view_0 (products.csv)
- $C_i$: file_0_view_0, column Capacity, key SectionID
- $v_j$: file_1_view_0, column Value, key ProductName
- $w_j$: file_1_view_0, column Weight, key ProductName
- $x_{ij}$: Decision variable for units of product $j$ in section $i$ (indexed by SectionID and ProductName)

All index sets, parameters, and constraints are mapped directly to the columns and keys as described above.