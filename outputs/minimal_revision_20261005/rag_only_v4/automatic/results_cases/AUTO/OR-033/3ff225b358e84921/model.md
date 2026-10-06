**Mathematical Optimization Model**

**Index Sets:**
- \( I \): Set of all products classified under ‘Baby’  
  (from all records in **file_0_view_0**, column **Product Name**)

**Parameters:**
- \( r_i \): Revenue per unit of product \( i \)  
  (from **file_0_view_0**, column **Revenue**)
- \( d_i \): Demand quantity for product \( i \)  
  (from **file_0_view_0**, column **Demand**)
- \( s_i \): Initial inventory for product \( i \)  
  (from **file_0_view_0**, column **Initial Inventory**)

**Decision Variables:**
- \( x_i \): Number of units of product \( i \) to fulfill  
  (domain: integer, \( 0 \leq x_i \leq \min\{d_i, s_i\} \), for all \( i \in I \))

**Objective:**
\[
\max \sum_{i \in I} r_i x_i
\]

**Constraints:**
1. **Inventory and Demand Fulfillment Bounds:**  
  For all \( i \in I \):
\[
0 \leq x_i \leq \min\{d_i, s_i\}
\]

**Data Mapping:**
- **Index set \( I \):** All records in **file_0_view_0**, column **Product Name**
- **Parameter \( r_i \):** **file_0_view_0**, column **Revenue**
- **Parameter \( d_i \):** **file_0_view_0**, column **Demand**
- **Parameter \( s_i \):** **file_0_view_0**, column **Initial Inventory**