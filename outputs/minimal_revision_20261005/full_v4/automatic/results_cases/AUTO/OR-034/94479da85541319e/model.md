#### Index Sets

- $I$: Set of all baked goods in the bakery, indexed by $i$.

#### Parameters

- $A_i$: Revenue per unit of baked good $i$.  
  (From column "Revenue" in table_id: file_0_view_0)
- $d_i$: Total demand for baked good $i$ over the sales horizon.  
  (From column "Demand" in table_id: file_0_view_0)
- $I_i$: Initial inventory available for baked good $i$.  
  (From column "Initial Inventory" in table_id: file_0_view_0)

#### Decision Variables

- $x_i$: Quantity of baked good $i$ to fulfill (integer, $0 \leq x_i \leq \min\{d_i, I_i\}$), for all $i \in I$.

#### Objective

$$
\max \sum_{i \in I} A_i \cdot x_i
$$

#### Constraints

1. **Demand Fulfillment Constraint:**  
   $$
   x_i \leq d_i \qquad \forall i \in I
   $$

2. **Inventory Constraint:**  
   $$
   x_i \leq I_i \qquad \forall i \in I
   $$

3. **Non-negativity and Integrality:**  
   $$
   x_i \in \mathbb{Z}_{+} \qquad \forall i \in I
   $$

---

#### Data Mapping

- **Index Set $I$:** All records in table_id: file_0_view_0, column "Product Name"
- **Parameter $A_i$:** table_id: file_0_view_0, column "Revenue"
- **Parameter $d_i$:** table_id: file_0_view_0, column "Demand"
- **Parameter $I_i$:** table_id: file_0_view_0, column "Initial Inventory"