#### Abstract Mathematical Model

**Sets:**
- $I$: Set of displays (indexed by $i$), corresponding to ShelfID in capacity.csv.
- $J$: Set of products (indexed by $j$), corresponding to ProductName in products.csv.

**Parameters:**
- $C_i$: Capacity of display $i$ (from Capacity column in capacity.csv, table_id: file_0_view_0).
- $v_j$: Value of product $j$ (from Value column in products.csv, table_id: file_1_view_0).
- $w_j$: Weight of product $j$ (from Weight column in products.csv, table_id: file_1_view_0).

**Decision Variables:**
- $x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of units of product $j$ placed on display $i$.

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}
\]

**Constraints:**
1. **Display Capacity Constraints:**  
   For each display $i \in I$,
   \[
   \sum_{j \in J} w_j \, x_{ij} \leq C_i
   \]
2. **Minimum Allocation of First Product:**  
   Let $j^*$ be the index of the first product in products.csv (ProductName at source_row 0, i.e., "Smartphone"):
   \[
   \sum_{i \in I} x_{i j^*} \geq 5
   \]
3. **Nonnegativity and Integrality:**  
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J
   \]

---

#### Data Mapping

- **Displays ($I$):**  
  Table: capacity.csv (table_id: file_0_view_0)  
  Identifier: ShelfID

- **Display Capacity ($C_i$):**  
  Table: capacity.csv (table_id: file_0_view_0)  
  Column: Capacity  
  Key: ShelfID

- **Products ($J$):**  
  Table: products.csv (table_id: file_1_view_0)  
  Identifier: ProductName

- **Product Value ($v_j$):**  
  Table: products.csv (table_id: file_1_view_0)  
  Column: Value  
  Key: ProductName

- **Product Weight ($w_j$):**  
  Table: products.csv (table_id: file_1_view_0)  
  Column: Weight  
  Key: ProductName

- **First Product ($j^*$):**  
  ProductName at source_row 0 in products.csv (table_id: file_1_view_0)

---

**All sets, parameters, and constraints are defined using the exact columns and identifiers from the retrieved data.**