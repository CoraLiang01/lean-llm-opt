#### Mathematical Optimization Model

**Index Sets:**
- $I$: Set of display areas (indexed by $i$), with elements given by all DisplayID in file_0_view_0.
- $J$: Set of boat types (indexed by $j$), with elements given by all ProductName in file_1_view_0.

**Parameters:**
- $c_i$: Capacity of display area $i$ (from Capacity column in file_0_view_0, indexed by DisplayID).
- $v_j$: Value of boat type $j$ (from Value column in file_1_view_0, indexed by ProductName).
- $w_j$: Size (Weight) of boat type $j$ (from Weight column in file_1_view_0, indexed by ProductName).

**Decision Variables:**
- $x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of boats of type $j$ to place in display area $i$.

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}
\]

**Constraints:**
1. **Display Area Capacity Constraints:**
   \[
   \sum_{j \in J} w_j \, x_{ij} \leq c_i \qquad \forall i \in I
   \]
2. **Nonnegativity and Integrality:**
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
   \]

---

#### Data Mapping

- $I$: All DisplayID in table_id file_0_view_0, column DisplayID.
- $J$: All ProductName in table_id file_1_view_0, column ProductName.
- $c_i$: file_0_view_0, column Capacity, indexed by DisplayID.
- $v_j$: file_1_view_0, column Value, indexed by ProductName.
- $w_j$: file_1_view_0, column Weight, indexed by ProductName.
- $x_{ij}$: Decision variable for each $(i, j) \in I \times J$.

All parameters and index sets are defined by the full set of rows in the respective columns of the returned tables. No data is omitted or synthesized.