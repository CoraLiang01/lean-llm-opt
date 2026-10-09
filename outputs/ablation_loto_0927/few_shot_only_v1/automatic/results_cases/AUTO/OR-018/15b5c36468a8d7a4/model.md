#### Index Sets

- $I$: Set of all products classified as ‘Baby’.  
  (Data Mapping: All records in table_id = "file_0_view_0" where "Product Name" contains "Baby")

#### Parameters

- $A_i$: Revenue per unit of product $i \in I$.  
  (Data Mapping: "file_0_view_0", column "Revenue")
- $d_i$: Total demand for product $i \in I$ over the sales horizon.  
  (Data Mapping: "file_0_view_0", column "Demand")
- $I_i$: Initial inventory of product $i \in I$.  
  (Data Mapping: "file_0_view_0", column "Initial Inventory")

#### Decision Variables

- $x_i$: Number of units of product $i \in I$ to fulfill.  
  Domain: $x_i \in \mathbb{Z}_+$ (non-negative integers)

#### Objective

\[
\max \quad \sum_{i \in I} A_i \cdot x_i
\]

#### Constraints

1. **Inventory Constraint:**  
  $\quad x_i \leq I_i \quad \forall i \in I$

2. **Demand Constraint:**  
  $\quad x_i \leq d_i \quad \forall i \in I$

3. **Non-negativity and Integrality:**  
  $\quad x_i \in \mathbb{Z}_+, \quad \forall i \in I$

---

#### Data Mapping

- All parameters and index sets are sourced from table_id = "file_0_view_0" (file: Salesdata.csv), using columns:
    - "Product Name" (to identify ‘Baby’ products for set $I$)
    - "Revenue" (for $A_i$)
    - "Demand" (for $d_i$)
    - "Initial Inventory" (for $I_i$)