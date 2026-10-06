**Abstract Mathematical Optimization Model**

**Index Sets:**
- \( I \): Set of car models classified under ‘FDK57’.  
  (Data: all records in table_id = file_0_view_0)

**Parameters:**
- \( r_i \): Revenue per unit for car model \( i \).  
  (Source: file_0_view_0, column: Revenue)
- \( d_i \): Demand quantity for car model \( i \).  
  (Source: file_0_view_0, column: Demand)
- \( s_i \): Initial inventory for car model \( i \).  
  (Source: file_0_view_0, column: Initial Inventory)

**Decision Variables:**
- \( x_i \): Quantity of car model \( i \) to fulfill (integer, \( 0 \leq x_i \leq \min\{d_i, s_i\} \)), for all \( i \in I \).

**Objective:**
\[
\max \sum_{i \in I} r_i x_i
\]
(Maximize total revenue from fulfilled sales.)

**Constraints:**
1. **Demand fulfillment:**  
  \( x_i \leq d_i \), for all \( i \in I \)
2. **Inventory limit:**  
  \( x_i \leq s_i \), for all \( i \in I \)
3. **Non-negativity and integrality:**  
  \( x_i \geq 0 \), \( x_i \) integer, for all \( i \in I \)

---

**Data Mapping**

- **Index Set \( I \):** All records in table_id = file_0_view_0 (filtered for ‘Product Name’ prefix ‘FDK57’)
- **Parameter \( r_i \):** file_0_view_0, column: Revenue
- **Parameter \( d_i \):** file_0_view_0, column: Demand
- **Parameter \( s_i \):** file_0_view_0, column: Initial Inventory