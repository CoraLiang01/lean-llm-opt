#### Index Sets

- $I$: Set of all pizza types (from column "Product Name" in table_id = file_0_view_0).

#### Parameters

- $A_i$: Revenue per unit of pizza type $i$ (from column "Revenue", table_id = file_0_view_0).
- $d_i$: Total demand for pizza type $i$ (from column "Demand", table_id = file_0_view_0).
- $I_i$: Initial inventory for pizza type $i$ (from column "Initial Inventory", table_id = file_0_view_0).

#### Decision Variables

- $x_i$: Number of units of pizza type $i$ to fulfill, $x_i \in \mathbb{Z}_+, \forall i \in I$.

#### Objective

$$
\max \sum_{i \in I} A_i x_i
$$

#### Constraints

1. **Inventory Constraints:**  
   $$
   x_i \leq I_i, \quad \forall i \in I
   $$

2. **Demand Constraints:**  
   $$
   x_i \leq d_i, \quad \forall i \in I
   $$

3. **Non-negativity and Integrality:**  
   $$
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   $$

---

#### Data Mapping

- **Index Set $I$:**  
  All unique values in column "Product Name" of table_id = file_0_view_0.

- **Parameter $A_i$:**  
  Value in column "Revenue" for pizza type $i$ in table_id = file_0_view_0.

- **Parameter $d_i$:**  
  Value in column "Demand" for pizza type $i$ in table_id = file_0_view_0.

- **Parameter $I_i$:**  
  Value in column "Initial Inventory" for pizza type $i$ in table_id = file_0_view_0.