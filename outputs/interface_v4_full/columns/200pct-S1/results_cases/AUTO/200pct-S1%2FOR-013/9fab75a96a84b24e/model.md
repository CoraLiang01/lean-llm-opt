## Abstract Mathematical Model

Let:
- $I$ = set of storage areas, indexed by $i$ (from StorageID in file_0_view_0)
- $J$ = set of air conditioner types, indexed by $j$ (from ProductName in file_1_view_0)

Parameters:
- $c_i$ = capacity of storage area $i$ (Capacity from file_0_view_0, key: StorageID)
- $v_j$ = value of air conditioner type $j$ (Value from file_1_view_0, key: ProductName)
- $w_j$ = size (Weight) of air conditioner type $j$ (Weight from file_1_view_0, key: ProductName)

Decision Variables:
- $x_{ij}$ = number of units of air conditioner type $j$ placed in storage area $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

### Objective
Maximize the total value of air conditioners allocated:
$$
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
$$

### Constraints

1. **Storage Area Capacity Constraints** (for each storage area $i$):
$$
\sum_{j \in J} w_j \cdot x_{ij} \leq c_i \quad \forall i \in I
$$

2. **Nonnegativity and Integrality**:
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J
$$

---

### Data Mapping

- $I$ (storage areas): StorageID from file_0_view_0 (capacity.csv)
- $c_i$: Capacity from file_0_view_0, key: StorageID
- $J$ (air conditioner types): ProductName from file_1_view_0 (products.csv)
- $v_j$: Value from file_1_view_0, key: ProductName
- $w_j$: Weight from file_1_view_0, key: ProductName

All parameters and index sets are to be taken directly from the referenced columns and table_ids above, preserving source order and identifiers.