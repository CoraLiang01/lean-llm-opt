#### Abstract Mathematical Model

Let:
- $I$ = set of display areas, indexed by $i$ (DisplayID from capacity.csv)
- $J$ = set of vessel types, indexed by $j$ (ProductName from products.csv)

Parameters:
- $c_i$ = capacity of display area $i$ (from capacity.csv)
- $v_j$ = value of vessel type $j$ (from products.csv)
- $w_j$ = size (weight) of vessel type $j$ (from products.csv)

Decision Variables:
- $x_{ij} \in \mathbb{Z}_{\geq 0}$: number of vessels of type $j$ to place in display area $i$

Objective:
$$
\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}
$$

Subject to:
1. Capacity constraints for each display area:
$$
\sum_{j \in J} w_j \, x_{ij} \leq c_i \quad \forall i \in I
$$

2. Nonnegativity and integrality:
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J
$$

---

#### Data Mapping

- $I$ (Display Areas): file_0_view_0.DisplayID
- $J$ (Vessel Types): file_1_view_0.ProductName
- $c_i$: file_0_view_0, column Capacity, indexed by DisplayID
- $v_j$: file_1_view_0, column Value, indexed by ProductName
- $w_j$: file_1_view_0, column Weight, indexed by ProductName

Each $x_{ij}$ is the number of vessels of type $j$ to be placed in display area $i$. All parameters and index sets are mapped directly from the supplied files and columns.