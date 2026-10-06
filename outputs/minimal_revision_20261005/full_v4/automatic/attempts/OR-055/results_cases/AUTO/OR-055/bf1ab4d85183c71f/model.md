**Abstract Mathematical Model**

**Index Sets:**
- $I$: set of display areas, indexed by $i$ (from all DisplayID in file_0_view_0)
- $J$: set of boat types, indexed by $j$ (from all ProductName in file_1_view_0)

**Parameters:**
- $c_i$: capacity of display area $i$ (Capacity from file_0_view_0, indexed by DisplayID)
- $v_j$: value of one unit of boat type $j$ (Value from file_1_view_0, indexed by ProductName)
- $w_j$: size of one unit of boat type $j$ (Weight from file_1_view_0, indexed by ProductName)

**Decision Variables:**
- $x_{ij}$: number of units of boat type $j$ to place in display area $i$  
  Domain: $x_{ij} \in \mathbb{Z}_{\geq 0}$ (nonnegative integers), for all $i \in I$, $j \in J$

---

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}
\]

**Subject to:**

1. **Display Area Capacity Constraints:**  
 For each display area $i \in I$,
\[
\sum_{j \in J} w_j \, x_{ij} \leq c_i
\]

2. **Integrality and Nonnegativity:**  
\[
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J
\]

---

**Data Mapping**

- $I$: All DisplayID from `file_0_view_0` (capacity.csv), column `DisplayID`
- $J$: All ProductName from `file_1_view_0` (products.csv), column `ProductName`
- $c_i$: `file_0_view_0`, columns `DisplayID`, `Capacity`
- $v_j$: `file_1_view_0`, columns `ProductName`, `Value`
- $w_j$: `file_1_view_0`, columns `ProductName`, `Weight`