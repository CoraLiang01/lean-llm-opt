#### Index Sets

- $I$: Set of products, indexed by $i$.

#### Parameters

- $A_i$: Revenue per unit of product $i$.  
  (Data: table_id = file_0_view_0, column = "Revenue")
- $d_i$: Total demand for product $i$ over the sales horizon.  
  (Data: table_id = file_0_view_0, column = "Demand")
- $I_i$: Initial inventory of product $i$.  
  (Data: table_id = file_0_view_0, column = "Initial Inventory")

#### Decision Variables

- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$ (non-negative integers), for all $i \in I$.

#### Objective

\[
\max \quad \sum_{i \in I} A_i \, x_i
\]

#### Constraints

1. **Demand fulfillment cannot exceed demand:**
   \[
   x_i \leq d_i \qquad \forall i \in I
   \]

2. **Demand fulfillment cannot exceed initial inventory:**
   \[
   x_i \leq I_i \qquad \forall i \in I
   \]

3. **Non-negativity and integrality:**
   \[
   x_i \in \mathbb{Z}_+, \qquad \forall i \in I
   \]

---

#### Data Mapping

- **Index set $I$:** All product records in table_id = file_0_view_0, column = "Product Name"
- **Parameter $A_i$:** table_id = file_0_view_0, column = "Revenue"
- **Parameter $d_i$:** table_id = file_0_view_0, column = "Demand"
- **Parameter $I_i$:** table_id = file_0_view_0, column = "Initial Inventory"