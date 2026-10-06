**Abstract Mathematical Model**

**Index Sets:**
- \( I \): Set of all products, indexed by \( i \).  
  (Source: all "Product Name" in table_id = file_0_view_0)

**Parameters:**
- \( r_i \): Revenue per unit of product \( i \).  
  (Source: "Revenue", table_id = file_0_view_0)
- \( d_i \): Demand for product \( i \).  
  (Source: "Demand", table_id = file_0_view_0)
- \( s_i \): Initial inventory available for product \( i \).  
  (Source: "Initial Inventory", table_id = file_0_view_0)

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

- \( I \): All "Product Name" in table_id = file_0_view_0
- \( r_i \): "Revenue" in table_id = file_0_view_0
- \( d_i \): "Demand" in table_id = file_0_view_0
- \( s_i \): "Initial Inventory" in table_id = file_0_view_0