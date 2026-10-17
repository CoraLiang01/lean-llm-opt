#### Abstract Optimization Model

**Index Sets:**

- $I$ : Set of all products $i$ such that the value in column ‘Product Name’ of table_id file_0_view_0 contains the substring ‘Baby’.

**Parameters:**

- $A_i$ : Revenue per unit of product $i$, from column ‘Revenue’ in table_id file_0_view_0.
- $d_i$ : Demand for product $i$, from column ‘Demand’ in table_id file_0_view_0.
- $I_i$ : Initial inventory for product $i$, from column ‘Initial Inventory’ in table_id file_0_view_0.

**Decision Variables:**

- $x_i$ : Number of units of product $i$ to fulfill, for $i \in I$.

**Objective:**

$$
\max \sum_{i \in I} A_i \cdot x_i
$$

**Constraints:**

1. **Demand Constraint:** 
   $$
   x_i \leq d_i \quad \forall i \in I
   $$
2. **Inventory Constraint:** 
   $$
   x_i \leq I_i \quad \forall i \in I
   $$
3. **Nonnegativity and Integrality:** 
   $$
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   $$

---

#### Data Mapping

- **Source Table:** file_0_view_0 (EuropeSalesRecords.csv)
- **Index Set $I$:** All rows where ‘Product Name’ contains the substring ‘Baby’.
- **Parameter $A_i$:** Column ‘Revenue’ in file_0_view_0.
- **Parameter $d_i$:** Column ‘Demand’ in file_0_view_0.
- **Parameter $I_i$:** Column ‘Initial Inventory’ in file_0_view_0.