## Abstract Mathematical Model

Let:
- $I$ = set of storage areas, indexed by $i$ (from file_0_view_0, column StorageID)
- $J$ = set of air conditioner types, indexed by $j$ (from file_1_view_0, column ProductName)

Parameters:
- $c_i$ = capacity of storage area $i$ (file_0_view_0, column Capacity, key StorageID)
- $v_j$ = value of air conditioner type $j$ (file_1_view_0, column Value, key ProductName)
- $w_j$ = size (weight) of air conditioner type $j$ (file_1_view_0, column Weight, key ProductName)

Decision Variables:
- $x_{ij}$ = number of units of air conditioner type $j$ to place in storage area $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

### Objective
Maximize total value:
$$
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
$$

### Constraints

1. **Storage Area Capacity Constraints** (for each $i \in I$):
$$
\sum_{j \in J} w_j \cdot x_{ij} \leq c_i
$$

2. **Nonnegativity and Integrality** (for all $i \in I$, $j \in J$):
$$
x_{ij} \in \mathbb{Z}_{\geq 0}
$$

---

### Data Mapping

- $I$ (storage areas): file_0_view_0, column StorageID
- $c_i$: file_0_view_0, columns StorageID (key), Capacity (value)
- $J$ (air conditioner types): file_1_view_0, column ProductName
- $v_j$: file_1_view_0, columns ProductName (key), Value (value)
- $w_j$: file_1_view_0, columns ProductName (key), Weight (value)

All data is used as returned, preserving file and row order, and all constraints and variable domains are as described.