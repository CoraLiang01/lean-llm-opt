**Abstract Mathematical Model**

**Index Sets:**
- $I$: Set of products classified under ‘Books’, indexed by $i$ (from all records in file_0_view_0, column Product_Name).

**Parameters:**
- $r_i$: Revenue per unit of product $i$ (file_0_view_0, column Revenue).
- $d_i$: Demand for product $i$ (file_0_view_0, column Demand).
- $s_i$: Initial inventory of product $i$ (file_0_view_0, column Initial Inventory).

**Decision Variables:**
- $x_i$: Number of units of product $i$ to fulfill, $x_i \geq 0$, integer.

**Objective:**
\[
\max \sum_{i \in I} r_i x_i
\]

**Constraints:**
1. **Demand fulfillment:**  
   $\forall i \in I: \quad x_i \leq d_i$
2. **Inventory availability:**  
   $\forall i \in I: \quad x_i \leq s_i$
3. **Nonnegativity and integrality:**  
   $\forall i \in I: \quad x_i \in \mathbb{Z}_{\geq 0}$

---

**Data Mapping**

- $I$: All records in `file_0_view_0`, column `Product_Name` where `Product_Name` starts with "Books".
- $r_i$: `file_0_view_0`, column `Revenue`, keyed by `Product_Name`.
- $d_i$: `file_0_view_0`, column `Demand`, keyed by `Product_Name`.
- $s_i$: `file_0_view_0`, column `Initial Inventory`, keyed by `Product_Name`.