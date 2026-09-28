#### Abstract Mathematical Model

**Index Sets:**
- $I$: set of storage areas (indexed by $i$), from `capacity.csv` column `StorageID`
- $J$: set of air conditioner types (indexed by $j$), from `products.csv` column `ProductName`

**Parameters:**
- $C_i$: capacity of storage area $i$, from `capacity.csv` column `Capacity`
- $v_j$: value of one unit of air conditioner type $j$, from `products.csv` column `Value`
- $w_j$: size (weight) of one unit of air conditioner type $j$, from `products.csv` column `Weight`

**Decision Variables:**
- $x_{ij}$: number of units of air conditioner type $j$ to be placed in storage area $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
\]

**Constraints:**
1. **Storage Area Capacity Constraints:**
   \[
   \sum_{j \in J} w_j \cdot x_{ij} \leq C_i, \quad \forall i \in I
   \]
2. **Non-negativity and Integrality:**
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I,\, j \in J
   \]

---

#### Data Mapping

- $I$ (storage areas): `capacity.csv`, column `StorageID`
- $C_i$ (capacity): `capacity.csv`, column `Capacity`
- $J$ (air conditioner types): `products.csv`, column `ProductName`
- $v_j$ (value): `products.csv`, column `Value`
- $w_j$ (size/weight): `products.csv`, column `Weight`