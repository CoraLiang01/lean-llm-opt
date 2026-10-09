##### Mathematical Model

Let $I$ be the set of all products with Product_Name starting with "Books" (from table_id file_0_view_0).

**Decision Variables:**
- For each $i \in I$, let $x_i$ = number of units of product $i$ to fulfill. ($x_i \in \mathbb{Z}_{\geq 0}$)

**Parameters:**
- $r_i$ = Revenue per unit of product $i$ (from Revenue column)
- $d_i$ = Demand for product $i$ (from Demand column)
- $s_i$ = Initial Inventory for product $i$ (from Initial Inventory column)

**Objective:**
\[
\max \sum_{i \in I} r_i x_i
\]

**Constraints:**
1. **Demand fulfillment:**  
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
2. **Inventory availability:**  
   \[
   x_i \leq s_i \quad \forall i \in I
   \]
3. **Nonnegativity and integrality:**  
   \[
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
   \]

---

##### Data Mapping

- Index set $I$: All records in table_id file_0_view_0 where Product_Name starts with "Books"
- $r_i$: file_0_view_0, column "Revenue"
- $d_i$: file_0_view_0, column "Demand"
- $s_i$: file_0_view_0, column "Initial Inventory"
- $x_i$: Decision variable for each $i \in I$ (Books products)