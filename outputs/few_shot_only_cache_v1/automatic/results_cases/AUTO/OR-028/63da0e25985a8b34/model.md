#### Abstract Mathematical Optimization Model

**Index Sets:**
- $I$: Set of all products, indexed by $i$.

**Parameters:**
- $A_i$: Revenue per unit of product $i$.  
  (From column ‘Revenue’ in table_id = file_0_view_0)
- $d_i$: Total demand for product $i$ over the sales horizon.  
  (From column ‘Demand’ in table_id = file_0_view_0)
- $I_i$: Initial inventory available for product $i$.  
  (From column ‘Initial Inventory’ in table_id = file_0_view_0)

**Decision Variables:**
- $x_i$: Number of units of product $i$ to fulfill (integer, $x_i \geq 0$).

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. **Demand fulfillment cannot exceed demand:**
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
2. **Cannot fulfill more than available inventory:**
   \[
   x_i \leq I_i \quad \forall i \in I
   \]
3. **Non-negativity and integrality:**
   \[
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
   \]

---

#### Data Mapping

- **Index Set $I$:** All rows in table_id = file_0_view_0, column ‘Product Name’.
- **Parameter $A_i$:** table_id = file_0_view_0, column ‘Revenue’.
- **Parameter $d_i$:** table_id = file_0_view_0, column ‘Demand’.
- **Parameter $I_i$:** table_id = file_0_view_0, column ‘Initial Inventory’.