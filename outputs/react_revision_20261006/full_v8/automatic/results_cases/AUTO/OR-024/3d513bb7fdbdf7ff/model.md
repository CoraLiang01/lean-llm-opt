#### Symbolic Mathematical Model

Let:
- $I$ = index set of all products with names starting with "S700_" (from the data)
- For each $i \in I$:
    - $A_i$ = revenue per unit of product $i$ (parameter)
    - $d_i$ = deterministic demand for product $i$ (parameter)
    - $s_i$ = initial inventory for product $i$ (parameter)
    - $x_i$ = number of units of product $i$ to fulfill (decision variable)

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Subject to:**
1. **Inventory constraints:**
   \[
   x_i \leq s_i \quad \forall i \in I
   \]
2. **Demand constraints:**
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
3. **Nonnegativity and integrality:**
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

#### Data Mapping

- **Index set $I$:** All rows in table_id file_0_view_0 where "Product Name" has prefix "S700_"
- **Parameter $A_i$:** "Revenue" column in table_id file_0_view_0
- **Parameter $d_i$:** "Demand" column in table_id file_0_view_0
- **Parameter $s_i$:** "Initial Inventory" column in table_id file_0_view_0
- **Decision variable $x_i$:** Number of units to fulfill for each $i \in I$ (as defined above)