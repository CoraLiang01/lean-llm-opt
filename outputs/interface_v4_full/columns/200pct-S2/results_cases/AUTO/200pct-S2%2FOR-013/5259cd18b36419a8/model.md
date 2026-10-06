## Abstract Mathematical Model

Let:
- $I$ = set of storage areas, indexed by $i$, with StorageID from file_0_view_0 (capacity.csv)
- $J$ = set of air conditioner types, indexed by $j$, with ProductName from file_1_view_0 (products.csv)

Parameters:
- $c_i$ = capacity of storage area $i$ (Capacity, file_0_view_0, StorageID)
- $v_j$ = value per unit of air conditioner type $j$ (Value, file_1_view_0, ProductName)
- $w_j$ = size (Weight) per unit of air conditioner type $j$ (Weight, file_1_view_0, ProductName)

Decision Variables:
- $x_{ij}$ = number of units of air conditioner type $j$ to place in storage area $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

### Objective
Maximize total value of air conditioners allocated:
$$
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
$$

### Constraints

1. **Storage Area Capacity Constraints**  
For each storage area $i \in I$:
$$
\sum_{j \in J} w_j \cdot x_{ij} \leq c_i
$$

2. **Nonnegativity and Integrality**
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J
$$

---

### Data Mapping

- $I$ (storage areas): file_0_view_0, column StorageID
- $c_i$: file_0_view_0, columns StorageID (key), Capacity (value)
- $J$ (air conditioner types): file_1_view_0, column ProductName
- $v_j$: file_1_view_0, columns ProductName (key), Value (value)
- $w_j$: file_1_view_0, columns ProductName (key), Weight (value)
- $x_{ij}$: decision variable for each $(i,j)$ pair

All other columns in the source files are ignored for this model.