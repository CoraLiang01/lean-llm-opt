**Abstract Mathematical Optimization Model**

**Index Sets:**
- \( I \): Set of products, indexed by \( i \).  
  (Source: all unique values in "Product Name" from table_id = file_0_view_0)

**Parameters:**
- \( r_i \): Revenue per unit of product \( i \).  
  (Source: "Revenue" in file_0_view_0)
- \( d_i \): Demand for product \( i \) over the sales horizon.  
  (Source: "Demand" in file_0_view_0)
- \( s_i \): Initial inventory available for product \( i \).  
  (Source: "Initial Inventory" in file_0_view_0)

**Decision Variables:**
- \( x_i \): Number of units of product \( i \) to fulfill for customer purchases.  
  Domain: Integer, \( 0 \leq x_i \leq \min\{d_i, s_i\} \), for all \( i \in I \)

**Objective:**
\[
\max \sum_{i \in I} r_i x_i
\]
(Maximize total revenue from fulfilled sales.)

**Constraints:**
1. **Demand fulfillment constraint:**  
  \( x_i \leq d_i \), for all \( i \in I \)  
  (Cannot fulfill more than demand.)

2. **Inventory constraint:**  
  \( x_i \leq s_i \), for all \( i \in I \)  
  (Cannot fulfill more than available inventory.)

3. **Non-negativity and integrality:**  
  \( x_i \geq 0 \), \( x_i \) integer, for all \( i \in I \)

---

**Data Mapping**

- **Index Set \( I \):**  
  All "Product Name" entries from table_id = file_0_view_0

- **Parameter \( r_i \):**  
  "Revenue" column from table_id = file_0_view_0

- **Parameter \( d_i \):**  
  "Demand" column from table_id = file_0_view_0

- **Parameter \( s_i \):**  
  "Initial Inventory" column from table_id = file_0_view_0

- **Decision Variable \( x_i \):**  
  Defined for each \( i \in I \) as above

---

**Summary:**  
This model maximizes total revenue by optimally allocating available inventory to meet deterministic product demands, subject to inventory and demand limits for each product. All data is mapped directly from the specified columns in table_id = file_0_view_0.