**Abstract Mathematical Model**

**Index Sets:**
- $I$: Set of products (from `Salesofsummerclothes.csv`, column `Product Name`)

**Parameters:**
- $r_i$: Revenue per unit of product $i$ (`Revenue`, table_id: file_0_view_0)
- $d_i$: Demand for product $i$ (`Demand`, table_id: file_0_view_0)
- $s_i$: Initial inventory for product $i$ (`Initial Inventory`, table_id: file_0_view_0)

**Decision Variables:**
- $x_i$: Number of units of product $i$ to fulfill (integer, $x_i \geq 0$)

**Objective:**
\[
\max \sum_{i \in I} r_i x_i
\]

**Constraints:**
1. **Demand fulfillment:**  
   \[
   x_i \leq d_i \qquad \forall i \in I
   \]
2. **Inventory limit:**  
   \[
   x_i \leq s_i \qquad \forall i \in I
   \]
3. **Nonnegativity and integrality:**  
   \[
   x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I
   \]

---

**Data Mapping**

- $I$: All rows in `file_0_view_0`, column `Product Name`
- $r_i$: `file_0_view_0`, column `Revenue`
- $d_i$: `file_0_view_0`, column `Demand`
- $s_i$: `file_0_view_0`, column `Initial Inventory`