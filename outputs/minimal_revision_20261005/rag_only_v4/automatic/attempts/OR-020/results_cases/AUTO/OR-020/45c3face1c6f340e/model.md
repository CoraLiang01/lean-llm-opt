**Abstract Mathematical Optimization Model**

**Index Sets:**
- \( I \): Set of all products, indexed by \( i \).  
  (Source: all unique values in "Product Name" from table_id: file_0_view_0)

**Parameters:**
- \( r_i \): Revenue per unit of product \( i \).  
  (Source: "Revenue" in file_0_view_0, mapped by "Product Name")
- \( d_i \): Demand for product \( i \).  
  (Source: "Demand" in file_0_view_0, mapped by "Product Name")
- \( s_i \): Initial inventory available for product \( i \).  
  (Source: "Initial Inventory" in file_0_view_0, mapped by "Product Name")

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
2. **Inventory availability constraint:**  
  \[
  x_i \leq s_i \qquad \forall i \in I
  \]
3. **Non-negativity and integrality:**  
  \[
  x_i \in \mathbb{Z}_+, \qquad 0 \leq x_i \leq \min\{d_i, s_i\} \qquad \forall i \in I
  \]

---

**Data Mapping**

- **Index Set \( I \):**  
  All "Product Name" values from table_id: file_0_view_0

- **Parameter \( r_i \):**  
  "Revenue" column in table_id: file_0_view_0, mapped by "Product Name"

- **Parameter \( d_i \):**  
  "Demand" column in table_id: file_0_view_0, mapped by "Product Name"

- **Parameter \( s_i \):**  
  "Initial Inventory" column in table_id: file_0_view_0, mapped by "Product Name"