**Abstract Mathematical Model**

**Sets:**
- $I$: Set of display areas, indexed by $i$ (DisplayID from capacity.csv)
- $J$: Set of vessel types, indexed by $j$ (ProductName from products.csv)

**Parameters:**
- $c_i$: Capacity of display area $i$ (column Capacity in capacity.csv)
- $v_j$: Value of vessel type $j$ (column Value in products.csv)
- $w_j$: Size (Weight) of vessel type $j$ (column Weight in products.csv)

**Decision Variables:**
- $x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of vessels of type $j$ to place in display area $i$

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
\]

**Constraints:**
1. **Display Area Capacity Constraints:**  
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

- $I$: capacity.csv, column DisplayID
- $c_i$: capacity.csv, column Capacity, key DisplayID
- $J$: products.csv, column ProductName
- $v_j$: products.csv, column Value, key ProductName
- $w_j$: products.csv, column Weight, key ProductName