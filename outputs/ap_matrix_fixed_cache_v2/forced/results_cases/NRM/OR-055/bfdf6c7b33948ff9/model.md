#### Abstract Mathematical Model

**Index Sets:**
- $I$: set of display areas (from capacity.csv, column "DisplayID")
- $J$: set of boat types (from products.csv, column "ProductName")

**Parameters:**
- $C_i$: capacity of display area $i \in I$ (from capacity.csv, column "Capacity")
- $v_j$: value of one unit of boat type $j \in J$ (from products.csv, column "Value")
- $w_j$: size of one unit of boat type $j \in J$ (from products.csv, column "Weight")

**Decision Variables:**
- $x_{ij}$: integer number of units of boat type $j$ to place in display area $i$, for all $i \in I$, $j \in J$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
\]

**Constraints:**
1. **Display Area Capacity Constraints:**
   \[
   \sum_{j \in J} w_j \cdot x_{ij} \leq C_i, \quad \forall i \in I
   \]
2. **Nonnegativity and Integrality:**
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I,\, j \in J
   \]

---

#### Data Mapping

- **Display Areas ($I$), Capacity ($C_i$):**
  - Source: capacity.csv
  - Table ID: file_0_view_0
  - Columns: "DisplayID" (index set $I$), "Capacity" (parameter $C_i$)
- **Boat Types ($J$), Value ($v_j$), Size ($w_j$):**
  - Source: products.csv
  - Table ID: file_1_view_0
  - Columns: "ProductName" (index set $J$), "Value" (parameter $v_j$), "Weight" (parameter $w_j$)