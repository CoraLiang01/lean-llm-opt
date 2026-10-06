#### Abstract Mathematical Model

**Index Sets:**
- $I$: Set of display areas, indexed by $i$ (DisplayID from file_0_view_0)
- $J$: Set of boat types, indexed by $j$ (ProductName from file_1_view_0)

**Parameters:**
- $c_i$: Capacity of display area $i$ (Capacity from file_0_view_0, indexed by DisplayID)
- $v_j$: Value of one unit of boat type $j$ (Value from file_1_view_0, indexed by ProductName)
- $w_j$: Size of one unit of boat type $j$ (Weight from file_1_view_0, indexed by ProductName)

**Decision Variables:**
- $x_{ij}$: Number of units of boat type $j$ to place in display area $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
\]

**Constraints:**
1. **Display Area Capacity Constraints:**
   \[
   \sum_{j \in J} w_j \cdot x_{ij} \leq c_i \qquad \forall i \in I
   \]
2. **Integrality and Nonnegativity:**
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
   \]

---

#### Data Mapping

- $I$ (Display Areas): file_0_view_0, column DisplayID
- $c_i$: file_0_view_0, column Capacity, indexed by DisplayID
- $J$ (Boat Types): file_1_view_0, column ProductName
- $v_j$: file_1_view_0, column Value, indexed by ProductName
- $w_j$: file_1_view_0, column Weight, indexed by ProductName

All parameters and index sets are to be taken directly from the referenced columns and tables, preserving original order and identifiers.