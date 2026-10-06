#### Abstract Mathematical Model

**Index Sets:**
- $I$: Set of display areas (DisplayID from capacity.csv)
- $J$: Set of boat types (ProductName from products.csv)

**Parameters:**
- $c_i$: Capacity of display area $i$ (from Capacity column in capacity.csv)
- $v_j$: Value of one unit of boat type $j$ (from Value column in products.csv)
- $w_j$: Size (weight) of one unit of boat type $j$ (from Weight column in products.csv)

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

- $I$ (display areas): file_0_view_0.DisplayID
- $J$ (boat types): file_1_view_0.ProductName
- $c_i$: file_0_view_0[DisplayID=$i$].Capacity
- $v_j$: file_1_view_0[ProductName=$j$].Value
- $w_j$: file_1_view_0[ProductName=$j$].Weight

All parameters and index sets are mapped directly from the original files and columns as specified above.