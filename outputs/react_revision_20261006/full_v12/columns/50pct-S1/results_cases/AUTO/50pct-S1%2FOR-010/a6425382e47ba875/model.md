## Mathematical Model

Let:
- $I$ = set of sections (indexed by $i$), from file_0_view_0:SectionID
- $J$ = set of products (indexed by $j$), from file_1_view_0:ProductName

Parameters:
- $c_i$ = capacity of section $i$ (file_0_view_0:Capacity)
- $v_j$ = value (price) of product $j$ (file_1_view_0:Value)
- $w_j$ = space requirement of product $j$ (file_1_view_0:Weight)

Decision variables:
- $x_{ij}$ = number of units of product $j$ to stock in section $i$, $x_{ij} \in \mathbb{Z}_{\geq 0}$

### Objective
Maximize total revenue:
$$
\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}
$$

### Constraints

Section capacity constraints:
$$
\sum_{j \in J} w_j \, x_{ij} \leq c_i \qquad \forall i \in I
$$

Nonnegativity and integrality:
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
$$

---

## Data Mapping

- $I$: file_0_view_0:SectionID
- $J$: file_1_view_0:ProductName
- $c_i$: file_0_view_0:Capacity, keyed by SectionID $i$
- $v_j$: file_1_view_0:Value, keyed by ProductName $j$
- $w_j$: file_1_view_0:Weight, keyed by ProductName $j$
- $x_{ij}$: number of units of product $j$ in section $i$ (decision variable, integer, $\geq 0$)