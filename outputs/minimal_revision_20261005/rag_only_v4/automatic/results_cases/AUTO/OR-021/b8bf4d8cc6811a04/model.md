**Abstract Mathematical Optimization Model**

**Index Sets:**
- \( I \): Set of all products, indexed by \( i \).  
  (Data: all "Product Name" entries in table_id = file_0_view_0)

**Parameters:**
- \( r_i \): Revenue per unit of product \( i \).  
  (Data: "Revenue" in file_0_view_0)
- \( d_i \): Demand for product \( i \).  
  (Data: "Demand" in file_0_view_0)
- \( s_i \): Initial inventory available for product \( i \).  
  (Data: "Initial Inventory" in file_0_view_0)

**Decision Variables:**
- \( x_i \): Number of units of product \( i \) to fulfill (integer, \( 0 \leq x_i \leq \min\{d_i, s_i\} \)).

**Objective:**
\[
\max \sum_{i \in I} r_i x_i
\]
(Maximize total revenue from fulfilled units.)

**Constraints:**
1. **Demand fulfillment constraint:**  
  \[
  x_i \leq d_i \qquad \forall i \in I
  \]
2. **Inventory constraint:**  
  \[
  x_i \leq s_i \qquad \forall i \in I
  \]
3. **Non-negativity and integrality:**  
  \[
  x_i \geq 0,\quad x_i \in \mathbb{Z} \qquad \forall i \in I
  \]

---

**Data Mapping**

- **Index Set \( I \):** All "Product Name" in table_id = file_0_view_0
- **Parameter \( r_i \):** "Revenue" in table_id = file_0_view_0
- **Parameter \( d_i \):** "Demand" in table_id = file_0_view_0
- **Parameter \( s_i \):** "Initial Inventory" in table_id = file_0_view_0