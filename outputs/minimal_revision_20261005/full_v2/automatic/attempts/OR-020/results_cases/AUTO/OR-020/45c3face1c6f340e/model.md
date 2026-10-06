#### Index Sets

- $I$: Set of all products (indexed by $i$).

#### Parameters

- $A_i$: Revenue per unit of product $i$.  
  (from table_id: file_0_view_0, column: Revenue)
- $d_i$: Demand for product $i$.  
  (from table_id: file_0_view_0, column: Demand)
- $I_i$: Initial inventory of product $i$.  
  (from table_id: file_0_view_0, column: Initial Inventory)

#### Decision Variables

- $x_i$: Number of units of product $i$ to fulfill, $\forall i \in I$  
  ($x_i \in \mathbb{Z}_+,\, x_i \geq 0$)

#### Objective

$$
\max \sum_{i \in I} A_i \cdot x_i
$$

#### Constraints

1. **Demand fulfillment:**  
   $$
   x_i \leq d_i, \quad \forall i \in I
   $$
2. **Inventory availability:**  
   $$
   x_i \leq I_i, \quad \forall i \in I
   $$
3. **Non-negativity and integrality:**  
   $$
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   $$

---

#### Data Mapping

- **Index set $I$:** All records in table_id: file_0_view_0, column: Product Name
- **Parameter $A_i$:** table_id: file_0_view_0, column: Revenue
- **Parameter $d_i$:** table_id: file_0_view_0, column: Demand
- **Parameter $I_i$:** table_id: file_0_view_0, column: Initial Inventory