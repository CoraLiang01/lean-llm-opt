**Abstract Mathematical Model**

**Index Set:**
- $I$: Set of baked goods, with each $i \in I$ corresponding to a unique "Product Name" in Frenchbakerydailysales.csv.

**Parameters:**
- $r_i$: Revenue per unit of baked good $i$ ("Revenue" column, table_id: file_0_view_0).
- $d_i$: Demand for baked good $i$ ("Demand" column, table_id: file_0_view_0).
- $s_i$: Initial inventory of baked good $i$ ("Initial Inventory" column, table_id: file_0_view_0).

**Decision Variables:**
- $x_i$: Quantity of baked good $i$ to fulfill (integer, $x_i \geq 0$).

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

- $I$: All rows in table_id: file_0_view_0, column "Product Name"
- $r_i$: table_id: file_0_view_0, column "Revenue", keyed by "Product Name"
- $d_i$: table_id: file_0_view_0, column "Demand", keyed by "Product Name"
- $s_i$: table_id: file_0_view_0, column "Initial Inventory", keyed by "Product Name"
- $x_i$: Decision variable for each $i \in I$