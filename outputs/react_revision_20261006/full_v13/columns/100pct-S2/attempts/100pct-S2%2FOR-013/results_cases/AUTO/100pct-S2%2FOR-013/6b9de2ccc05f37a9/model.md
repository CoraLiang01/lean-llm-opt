### Mathematical Model

**Index Sets:**
- $I$: set of storage areas (indexed by $i$), with IDs from file_0_view_0.StorageID
- $J$: set of air conditioner types (indexed by $j$), with names from file_1_view_0.ProductName

**Parameters:**
- $c_i$: capacity of storage area $i$ (file_0_view_0.Capacity, indexed by StorageID)
- $v_j$: value per unit of air conditioner type $j$ (file_1_view_0.Value, indexed by ProductName)
- $w_j$: size per unit of air conditioner type $j$ (file_1_view_0.Weight, indexed by ProductName)

**Decision Variables:**
- $x_{ij}$: number of units of air conditioner type $j$ to place in storage area $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}
\]

**Constraints:**
1. **Storage Area Capacity:**
   \[
   \sum_{j \in J} w_j \, x_{ij} \leq c_i \qquad \forall i \in I
   \]
2. **Integrality and Nonnegativity:**
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
   \]

---

### Data Mapping

- $I$: file_0_view_0.StorageID
- $J$: file_1_view_0.ProductName
- $c_i$: file_0_view_0.Capacity (indexed by StorageID)
- $v_j$: file_1_view_0.Value (indexed by ProductName)
- $w_j$: file_1_view_0.Weight (indexed by ProductName)
- $x_{ij}$: number of units of product $j$ in storage area $i$ (decision variable, integer, $\geq 0$)