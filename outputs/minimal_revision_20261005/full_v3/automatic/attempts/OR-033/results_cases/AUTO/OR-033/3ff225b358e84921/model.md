#### Index Sets

- $I$: Set of all products classified under ‘Baby’ (from table_id: file_0_view_0, column: Product Name).

#### Parameters

- $A_i$: Revenue per unit of product $i$ (from table_id: file_0_view_0, column: Revenue).
- $d_i$: Total deterministic demand for product $i$ (from table_id: file_0_view_0, column: Demand).
- $I_i$: Initial inventory available for product $i$ (from table_id: file_0_view_0, column: Initial Inventory).

#### Decision Variables

- $x_i$: Number of units of product $i \in I$ to fulfill, integer, $x_i \geq 0$.

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
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
   $$

---

#### Data Mapping

- **Index Set $I$:**  
  - Source: table_id: file_0_view_0, column: Product Name

- **Parameter $A_i$:**  
  - Source: table_id: file_0_view_0, column: Revenue

- **Parameter $d_i$:**  
  - Source: table_id: file_0_view_0, column: Demand

- **Parameter $I_i$:**  
  - Source: table_id: file_0_view_0, column: Initial Inventory

- **Variable $x_i$:**  
  - Decision variable as defined in the query, for each $i \in I$