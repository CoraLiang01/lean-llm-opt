#### Index Sets

- $I$: Set of all products classified under ‘27in’ (from column "Product Name" in table_id: file_0_view_0).

#### Parameters

- $A_i$: Revenue per unit of product $i$ (from column "Revenue", table_id: file_0_view_0).
- $d_i$: Total demand for product $i$ (from column "Demand", table_id: file_0_view_0).
- $I_i$: Initial inventory for product $i$ (from column "Initial Inventory", table_id: file_0_view_0).

#### Decision Variables

- $x_i$: Number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+$.

#### Objective

$$
\max \sum_{i \in I} A_i \cdot x_i
$$

#### Constraints

1. **Inventory Constraint:**  
   $$
   x_i \leq I_i \quad \forall i \in I
   $$

2. **Demand Constraint:**  
   $$
   x_i \leq d_i \quad \forall i \in I
   $$

3. **Non-negativity and Integrality:**  
   $$
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   $$

---

#### Data Mapping

- **Index Set $I$:**  
  All records in table_id: file_0_view_0, column "Product Name", filtered by prefix "27in".

- **Parameter $A_i$:**  
  table_id: file_0_view_0, column "Revenue".

- **Parameter $d_i$:**  
  table_id: file_0_view_0, column "Demand".

- **Parameter $I_i$:**  
  table_id: file_0_view_0, column "Initial Inventory".