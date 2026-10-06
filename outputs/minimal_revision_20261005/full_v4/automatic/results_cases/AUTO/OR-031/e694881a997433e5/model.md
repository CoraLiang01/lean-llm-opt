#### Index Sets

- $I$: Set of all dairy products, indexed by $i$.

#### Parameters

- $A_i$: Revenue per unit of product $i$.  
  (From column "Revenue" in table_id: file_0_view_0)
- $d_i$: Total deterministic demand for product $i$.  
  (From column "Demand" in table_id: file_0_view_0)
- $I_i$: Initial inventory available for product $i$.  
  (From column "Initial Inventory" in table_id: file_0_view_0)

#### Decision Variables

- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$ (non-negative integers), for all $i \in I$.

#### Objective

$$
\max \sum_{i \in I} A_i \cdot x_i
$$

#### Constraints

1. **Inventory Constraints:**  
   $$
   x_i \leq I_i \qquad \forall i \in I
   $$

2. **Demand Constraints:**  
   $$
   x_i \leq d_i \qquad \forall i \in I
   $$

3. **Non-negativity and Integrality:**  
   $$
   x_i \in \mathbb{Z}_+, \qquad \forall i \in I
   $$

---

#### Data Mapping

- **Index Set $I$:** All records in table_id: file_0_view_0, column "Full_Product_Name"
- **Parameter $A_i$:** table_id: file_0_view_0, column "Revenue"
- **Parameter $d_i$:** table_id: file_0_view_0, column "Demand"
- **Parameter $I_i$:** table_id: file_0_view_0, column "Initial Inventory"