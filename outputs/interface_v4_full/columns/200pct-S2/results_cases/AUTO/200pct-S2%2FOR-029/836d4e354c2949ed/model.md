## Abstract Mathematical Model

**Sets:**
- $I$: set of displays (indexed by $i$), with business identifier $\texttt{ShelfID}$ from file_0_view_0 (capacity.csv)
- $J$: set of products (indexed by $j$), with business identifier $\texttt{ProductName}$ from file_1_view_0 (products.csv)

**Parameters:**
- $c_i$: capacity of display $i$ ($\texttt{Capacity}$ from file_0_view_0, indexed by $\texttt{ShelfID}$)
- $v_j$: value per unit of product $j$ ($\texttt{Value}$ from file_1_view_0, indexed by $\texttt{ProductName}$)
- $w_j$: weight per unit of product $j$ ($\texttt{Weight}$ from file_1_view_0, indexed by $\texttt{ProductName}$)

**Decision Variables:**
- $x_{ij} \in \mathbb{Z}_{\geq 0}$: number of units of product $j$ placed on display $i$

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}
\]

**Constraints:**
1. **Display Capacity Constraints:**  
   For each display $i \in I$,
   \[
   \sum_{j \in J} w_j \, x_{ij} \leq c_i
   \]
2. **Minimum Allocation of First Product:**  
   Let $j^*$ be the first product in file_1_view_0 (products.csv, $\texttt{source_row}=0$):
   \[
   \sum_{i \in I} x_{i j^*} \geq 5
   \]
3. **Nonnegativity and Integrality:**  
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J
   \]

---

## Data Mapping

**file_0_view_0 (capacity.csv):**
- Display index: $\texttt{ShelfID}$
- Capacity: $\texttt{Capacity}$

**file_1_view_0 (products.csv):**
- Product index: $\texttt{ProductName}$
- Value: $\texttt{Value}$
- Weight: $\texttt{Weight}$

**Variable:** $x_{ij}$ is indexed by $(\texttt{ShelfID}, \texttt{ProductName})$.

**Special constraint:** The "first product" is the record with $\texttt{source_row}=0$ in file_1_view_0.

---

**All parameters and indices are mapped directly from the original files and columns as described above.**