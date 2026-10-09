##### Mathematical Model

Let:
- $I$ = set of sections, indexed by $i$ (SectionID from file_0_view_0)
- $J$ = set of products, indexed by $j$ (ProductName from file_1_view_0)
- $x_{ij}$ = number of units of product $j$ to be placed in section $i$ (decision variable, integer, $\geq 0$)
- $v_j$ = value (revenue) per unit of product $j$ (Value from file_1_view_0)
- $w_j$ = space requirement per unit of product $j$ (Weight from file_1_view_0)
- $C_i$ = total display space capacity of section $i$ (Capacity from file_0_view_0)

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}
\]

**Subject to:**

- Section capacity constraints (for each section $i$):
\[
\sum_{j \in J} w_j \, x_{ij} \leq C_i \qquad \forall i \in I
\]

- Nonnegativity and integrality:
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
\]

---

##### Data Mapping

- $I$: SectionID from file_0_view_0 (capacity.csv)
- $J$: ProductName from file_1_view_0 (products.csv)
- $C_i$: file_0_view_0, column Capacity, keyed by SectionID
- $v_j$: file_1_view_0, column Value, keyed by ProductName
- $w_j$: file_1_view_0, column Weight, keyed by ProductName
- $x_{ij}$: decision variable for units of product $j$ in section $i$ (indexed by SectionID and ProductName)

All sections and products from the returned data are included. All variables are nonnegative integers. Every section's total product space used cannot exceed its Capacity. The objective is to maximize total revenue.