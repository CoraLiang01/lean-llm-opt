**Sets:**  
- $I$: Index set of all products classified as ‘ELE-S’ in table_id = file_0_view_0, column Product_Reference.

**Parameters:**  
- $A_i$: Revenue per unit of product $i \in I$ (from table_id = file_0_view_0, column Revenue).
- $d_i$: Demand for product $i \in I$ (from table_id = file_0_view_0, column Demand).
- $I_i$: Initial inventory for product $i \in I$ (from table_id = file_0_view_0, column Initial Inventory).

**Decision Variables:**  
- $x_i$: Number of units of product $i \in I$ to fulfill, integer, $0 \leq x_i \leq \min\{d_i, I_i\}$.

**Objective:**  
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**  
1. **Inventory and Demand Fulfillment:**  
   \[
   0 \leq x_i \leq \min\{d_i, I_i\} \qquad \forall i \in I
   \]
   (Or equivalently, two separate constraints:)
   \[
   x_i \leq d_i \qquad \forall i \in I
   \]
   \[
   x_i \leq I_i \qquad \forall i \in I
   \]
   \[
   x_i \geq 0, \quad x_i \in \mathbb{Z} \qquad \forall i \in I
   \]

**Data Mapping:**  
- All parameters and index sets are drawn from table_id = file_0_view_0 (file: SalesStoreoverview.csv), using columns:
    - Product_Reference (for set $I$ and product identification)
    - Revenue (for $A_i$)
    - Demand (for $d_i$)
    - Initial Inventory (for $I_i$)

**End of Model.**