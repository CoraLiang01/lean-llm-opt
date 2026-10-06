**Abstract Mathematical Model**

**Sets:**
- $I$: set of display areas, indexed by $i$ (DisplayID from file_0_view_0)
- $J$: set of boat types, indexed by $j$ (ProductName from file_1_view_0)

**Parameters:**
- $c_i$: capacity of display area $i$ (Capacity from file_0_view_0)
- $v_j$: value of boat type $j$ (Value from file_1_view_0)
- $w_j$: size (weight) of boat type $j$ (Weight from file_1_view_0)

**Decision Variables:**
- $x_{ij}$: number of boats of type $j$ to place in display area $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
\]

**Constraints:**
1. **Display Area Capacity:**  
   For each display area $i \in I$,
   \[
   \sum_{j \in J} w_j \cdot x_{ij} \leq c_i
   \]
2. **Nonnegativity and Integrality:**  
   For all $i \in I$, $j \in J$,
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0}
   \]

---

**Data Mapping**

- $I$: file_0_view_0, column DisplayID
- $J$: file_1_view_0, column ProductName
- $c_i$: file_0_view_0, column Capacity, keyed by DisplayID
- $v_j$: file_1_view_0, column Value, keyed by ProductName
- $w_j$: file_1_view_0, column Weight, keyed by ProductName
- $x_{ij}$: decision variable, for each $(i,j) \in I \times J$