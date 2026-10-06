**Abstract Mathematical Model**

**Sets:**
- $I$: Set of display areas, indexed by $i$ (DisplayID from file_0_view_0)
- $J$: Set of boat types, indexed by $j$ (ProductName from file_1_view_0)

**Parameters:**
- $c_i$: Capacity of display area $i$ (Capacity from file_0_view_0, indexed by DisplayID)
- $v_j$: Value of boat type $j$ (Value from file_1_view_0, indexed by ProductName)
- $w_j$: Size of boat type $j$ (Weight from file_1_view_0, indexed by ProductName)

**Decision Variables:**
- $x_{ij}$: Number of units of boat type $j$ to place in display area $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

---

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
\]

**Subject to:**

1. **Display Area Capacity Constraints:**
   \[
   \sum_{j \in J} w_j \cdot x_{ij} \leq c_i \qquad \forall i \in I
   \]

2. **Integrality and Nonnegativity:**
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
   \]

---

**Data Mapping**

- $I$: file_0_view_0.DisplayID
- $J$: file_1_view_0.ProductName
- $c_i$: file_0_view_0.Capacity, indexed by DisplayID
- $v_j$: file_1_view_0.Value, indexed by ProductName
- $w_j$: file_1_view_0.Weight, indexed by ProductName

- $x_{ij}$: Number of units of boat type $j$ in display area $i$ (decision variable, integer, $\geq 0$)