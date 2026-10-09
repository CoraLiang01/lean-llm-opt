### Mathematical Model

**Sets:**
- $I$: set of display areas (indexed by $i$), with identifiers from `file_0_view_0.DisplayID`
- $J$: set of boat types (indexed by $j$), with identifiers from `file_1_view_0.ProductName`

**Parameters:**
- $c_i$: capacity of display area $i$ (`file_0_view_0.Capacity`)
- $v_j$: value of one unit of boat type $j$ (`file_1_view_0.Value`)
- $w_j$: size (weight) of one unit of boat type $j$ (`file_1_view_0.Weight`)

**Decision Variables:**
- $x_{ij}$: number of boats of type $j$ to place in display area $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}
\]

**Constraints:**
1. **Display Area Capacity:**
   \[
   \sum_{j \in J} w_j \, x_{ij} \leq c_i \qquad \forall i \in I
   \]
2. **Nonnegativity and Integrality:**
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
   \]

---

### Data Mapping

- $I$: All `DisplayID` in `file_0_view_0` (from `capacity.csv`)
- $J$: All `ProductName` in `file_1_view_0` (from `products.csv`)
- $c_i$: `file_0_view_0.Capacity` for display area $i$
- $v_j$: `file_1_view_0.Value` for boat type $j$
- $w_j$: `file_1_view_0.Weight` for boat type $j$
- $x_{ij}$: Number of boats of type $j$ assigned to display area $i$ (decision variable)