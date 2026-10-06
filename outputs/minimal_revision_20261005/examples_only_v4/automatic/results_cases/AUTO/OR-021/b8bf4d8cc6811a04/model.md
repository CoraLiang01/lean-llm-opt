**Abstract Mathematical Optimization Model**

**Index Sets:**
- \( I \): Set of all products, indexed by \( i \).  
  (Data: all "Product Name" entries in table_id = file_0_view_0)

**Parameters:**
- \( r_i \): Revenue per unit of product \( i \).  
  (Data: "Revenue", table_id = file_0_view_0, column = "Revenue")
- \( d_i \): Demand for product \( i \).  
  (Data: "Demand", table_id = file_0_view_0, column = "Demand")
- \( s_i \): Initial inventory available for product \( i \).  
  (Data: "Initial Inventory", table_id = file_0_view_0, column = "Initial Inventory")

**Decision Variables:**
- \( x_i \): Number of units of product \( i \) to fulfill (integer, \( 0 \leq x_i \leq \min\{d_i, s_i\} \))

**Objective:**
\[
\max \sum_{i \in I} r_i x_i
\]

**Constraints:**
1. **Demand fulfillment constraint:**  
  \( x_i \leq d_i \quad \forall i \in I \)
2. **Inventory constraint:**  
  \( x_i \leq s_i \quad \forall i \in I \)
3. **Non-negativity and integrality:**  
  \( x_i \geq 0 \) and integer \( \forall i \in I \)

---

**Data Mapping**

- \( I \): All records in table_id = file_0_view_0, column = "Product Name"
- \( r_i \): table_id = file_0_view_0, column = "Revenue"
- \( d_i \): table_id = file_0_view_0, column = "Demand"
- \( s_i \): table_id = file_0_view_0, column = "Initial Inventory"