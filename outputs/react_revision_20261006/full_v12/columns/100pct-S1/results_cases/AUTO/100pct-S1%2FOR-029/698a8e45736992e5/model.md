### Mathematical Model

**Index Sets:**
- $I$: set of displays (from file_0_view_0, column ShelfID)
- $J$: set of products (from file_1_view_0, column ProductName)

**Parameters:**
- $c_i$: capacity of display $i$ (from file_0_view_0, column Capacity, key ShelfID)
- $v_j$: value per unit of product $j$ (from file_1_view_0, column Value, key ProductName)
- $w_j$: weight per unit of product $j$ (from file_1_view_0, column Weight, key ProductName)

**Decision Variables:**
- $x_{ij} \in \mathbb{Z}_{\geq 0}$: number of units of product $j$ placed on display $i$

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}
\]

**Constraints:**
1. **Display Capacity Constraints:**
   \[
   \sum_{j \in J} w_j \, x_{ij} \leq c_i \qquad \forall i \in I
   \]
2. **Minimum Placement of First Product:**
   \[
   \sum_{i \in I} x_{i j^*} \geq 5
   \]
   where $j^*$ is the ProductName in the first row of file_1_view_0.
3. **Nonnegativity and Integrality:**
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
   \]

---

### Data Mapping

- $I$: All ShelfID in file_0_view_0 (capacity.csv), column ShelfID
- $J$: All ProductName in file_1_view_0 (products.csv), column ProductName
- $c_i$: file_0_view_0, column Capacity, key ShelfID
- $v_j$: file_1_view_0, column Value, key ProductName
- $w_j$: file_1_view_0, column Weight, key ProductName
- $j^*$: ProductName in file_1_view_0, source_row 0

All variables, parameters, and constraints are indexed and mapped exactly as above.