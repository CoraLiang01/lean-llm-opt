#### Abstract Optimization Model

**Index Sets:**
- $I$: set of display areas (from `capacity.csv`, column `DisplayID`)
- $J$: set of vessel types (from `products.csv`, column `ProductName`)

**Parameters:**
- $C_i$: capacity of display area $i \in I$ (from `capacity.csv`, column `Capacity`)
- $v_j$: value of vessel type $j \in J$ (from `products.csv`, column `Value`)
- $w_j$: size (weight/dimension) of vessel type $j \in J$ (from `products.csv`, column `Weight`)

**Decision Variables:**
- $x_{ij}$: number of vessels of type $j \in J$ assigned to display area $i \in I$, where $x_{ij} \in \mathbb{Z}_{\geq 0}$

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
\]

**Constraints:**

1. **Display Area Capacity Constraints:**
   \[
   \sum_{j \in J} w_j \cdot x_{ij} \leq C_i \qquad \forall i \in I
   \]

2. **Non-negativity and Integrality:**
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
   \]

---

#### Data Mapping

- $I$ (display areas): `capacity.csv`, column `DisplayID`
- $C_i$ (display area capacities): `capacity.csv`, column `Capacity`
- $J$ (vessel types): `products.csv`, column `ProductName`
- $v_j$ (vessel values): `products.csv`, column `Value`
- $w_j$ (vessel sizes): `products.csv`, column `Weight`
- $x_{ij}$: number of vessels of type $j$ assigned to display area $i$ (decision variable, not in source data)

**Source Tables Used:**
- `capacity.csv` [columns: `DisplayID`, `Capacity`]
- `products.csv` [columns: `ProductName`, `Value`, `Weight`]