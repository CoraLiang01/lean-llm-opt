### Mathematical Model

Let:
- $I$ = set of cabinets, indexed by $i$ (from CabinetID in file_0_view_0)
- $J$ = set of coffee products, indexed by $j$ (from ProductName in file_1_view_0)
- $c_i$ = capacity of cabinet $i$ (Capacity from file_0_view_0)
- $v_j$ = value per unit of product $j$ (Value from file_1_view_0)
- $w_j$ = weight per unit of product $j$ (Weight from file_1_view_0)
- $x_{ij}$ = number of units of product $j$ placed in cabinet $i$ (decision variable, integer, $\geq 0$)

#### Objective:
Maximize total value:
$$
\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}
$$

#### Constraints:
- Cabinet capacity for each $i \in I$:
$$
\sum_{j \in J} w_j \, x_{ij} \leq c_i \qquad \forall i \in I
$$

- Nonnegativity and integrality:
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
$$

---

### Data Mapping

- $I$: CabinetID from file_0_view_0 (capacity.csv)
- $J$: ProductName from file_1_view_0 (products.csv)
- $c_i$: Capacity from file_0_view_0, column "Capacity", indexed by CabinetID
- $v_j$: Value from file_1_view_0, column "Value", indexed by ProductName
- $w_j$: Weight from file_1_view_0, column "Weight", indexed by ProductName
- $x_{ij}$: Number of units of product $j$ in cabinet $i$ (decision variable, integer, $\geq 0$)