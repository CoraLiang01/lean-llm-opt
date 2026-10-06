**Abstract Mathematical Model**

**Index Sets:**
- $I$: set of display areas (DisplayID from file_0_view_0)
- $J$: set of boat types (ProductName from file_1_view_0)

**Parameters:**
- $c_i$: capacity of display area $i$ (Capacity from file_0_view_0, indexed by DisplayID)
- $v_j$: value of boat type $j$ (Value from file_1_view_0, indexed by ProductName)
- $w_j$: size (Weight) of boat type $j$ (Weight from file_1_view_0, indexed by ProductName)

**Decision Variables:**
- $x_{ij}$: number of boats of type $j$ to place in display area $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

---

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}
\]

**Subject to:**

1. **Display Area Capacity Constraints:**
   \[
   \sum_{j \in J} w_j \, x_{ij} \leq c_i \qquad \forall i \in I
   \]

2. **Nonnegativity and Integrality:**
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
   \]

---

**Data Mapping**

- $I$: DisplayID from `file_0_view_0` (capacity.csv)
- $J$: ProductName from `file_1_view_0` (products.csv)
- $c_i$: `file_0_view_0`, column `Capacity`, key `DisplayID`
- $v_j$: `file_1_view_0`, column `Value`, key `ProductName`
- $w_j$: `file_1_view_0`, column `Weight`, key `ProductName`