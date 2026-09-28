#### Abstract Mathematical Model

**Index Sets:**
- $I$: set of display areas (indexed by $i$), from `capacity.csv` column `DisplayID`
- $J$: set of boat types (indexed by $j$), from `products.csv` column `ProductName`

**Parameters:**
- $C_i$: capacity of display area $i$, from `capacity.csv` column `Capacity`
- $v_j$: value of one unit of boat type $j$, from `products.csv` column `Value`
- $s_j$: size of one unit of boat type $j$, from `products.csv` column `Weight`

**Decision Variables:**
- $x_{ij}$: number of units of boat type $j$ to place in display area $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
\]

**Constraints:**

1. **Capacity constraints for each display area:**
   \[
   \sum_{j \in J} s_j \cdot x_{ij} \leq C_i \qquad \forall i \in I
   \]

2. **Non-negativity and integrality:**
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
   \]

---

#### Data Mapping

- $I$ (display areas): `capacity.csv`, column `DisplayID`
- $C_i$: `capacity.csv`, column `Capacity`
- $J$ (boat types): `products.csv`, column `ProductName`
- $v_j$: `products.csv`, column `Value`
- $s_j$: `products.csv`, column `Weight`
- $x_{ij}$: decision variable, number of units of boat type $j$ in display area $i$ (no direct data column; defined by the model)