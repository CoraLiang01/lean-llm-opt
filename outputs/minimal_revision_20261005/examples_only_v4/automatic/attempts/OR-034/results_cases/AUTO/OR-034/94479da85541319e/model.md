**Abstract Mathematical Optimization Model**

**Index Sets**
- \( I \): Set of all baked goods (indexed by \( i \)), where \( I = \) all unique values in [file_0_view_0, Product Name].

**Parameters**
- \( r_i \): Revenue per unit of baked good \( i \), from [file_0_view_0, Revenue].
- \( d_i \): Demand for baked good \( i \), from [file_0_view_0, Demand].
- \( s_i \): Initial inventory available for baked good \( i \), from [file_0_view_0, Initial Inventory].

**Decision Variables**
- \( x_i \): Quantity of baked good \( i \) to fulfill (units), \( x_i \geq 0 \), integer or continuous as appropriate.

**Objective**
\[
\max \sum_{i \in I} r_i x_i
\]
(Maximize total revenue from fulfilled sales.)

**Constraints**
1. **Demand fulfillment constraint:**  
   For all \( i \in I \):
   \[
   x_i \leq d_i
   \]
   (Cannot fulfill more than demand.)

2. **Inventory constraint:**  
   For all \( i \in I \):
   \[
   x_i \leq s_i
   \]
   (Cannot fulfill more than available inventory.)

3. **Non-negativity:**  
   For all \( i \in I \):
   \[
   x_i \geq 0
   \]

**Data Mapping**
- [file_0_view_0, Product Name] → Index set \( I \)
- [file_0_view_0, Revenue] → Parameter \( r_i \)
- [file_0_view_0, Demand] → Parameter \( d_i \)
- [file_0_view_0, Initial Inventory] → Parameter \( s_i \)