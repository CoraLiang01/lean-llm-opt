#### Index Sets

- $I$: Set of all car models with Product Name prefix ‘FDK57’ (from table_id: file_0_view_0).

#### Parameters

- $A_i$: Revenue per unit for car model $i \in I$ (from column: Revenue).
- $d_i$: Total demand for car model $i \in I$ (from column: Demand).
- $I_i$: Initial inventory for car model $i \in I$ (from column: Initial Inventory).

#### Decision Variables

- $x_i$: Number of units of car model $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+$.

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

3. **Nonnegativity and Integrality:**  
   $$
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   $$

---

#### Data Mapping

- **table_id:** file_0_view_0 (from BigMartSales.csv)
    - **Product Name:** Used to define index set $I$ (all rows where Product Name prefix is ‘FDK57’)
    - **Revenue:** Parameter $A_i$
    - **Demand:** Parameter $d_i$
    - **Initial Inventory:** Parameter $I_i$