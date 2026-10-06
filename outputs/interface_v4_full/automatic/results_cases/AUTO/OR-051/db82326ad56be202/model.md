## Abstract Mathematical Model

**Index Sets:**
- $I$: Set of cabinets, indexed by $i$ (from file_0_view_0, column CabinetID)
- $J$: Set of coffee products, indexed by $j$ (from file_1_view_0, column ProductName)

**Parameters:**
- $c_i$: Capacity of cabinet $i$ (from file_0_view_0, column Capacity, key CabinetID)
- $v_j$: Value per unit of product $j$ (from file_1_view_0, column Value, key ProductName)
- $w_j$: Weight per unit of product $j$ (from file_1_view_0, column Weight, key ProductName)

**Decision Variables:**
- $x_{ij}$: Number of units of product $j$ to place in cabinet $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
\]

**Constraints:**
1. **Cabinet Capacity Constraints:**  
   For each cabinet $i \in I$,
   \[
   \sum_{j \in J} w_j \cdot x_{ij} \leq c_i
   \]
2. **Integrality and Nonnegativity:**
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J
   \]

---

## Data Mapping

- $I$ (Cabinet set): file_0_view_0, column CabinetID
- $c_i$: file_0_view_0, columns CabinetID (key), Capacity (value)
- $J$ (Product set): file_1_view_0, column ProductName
- $v_j$: file_1_view_0, columns ProductName (key), Value (value)
- $w_j$: file_1_view_0, columns ProductName (key), Weight (value)
- $x_{ij}$: Decision variable for each $(i,j)$ pair

All data is used as returned, with no omitted records or synthesized identifiers.