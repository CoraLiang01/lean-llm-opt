#### Abstract Mathematical Model

**Index Set:**  
Let $I$ be the set of all products with identifiers beginning with ‘S700_’ in the data.

**Parameters:**  
For each $i \in I$:
- $A_i$: Revenue per unit of product $i$ (from column ‘Revenue’)
- $d_i$: Demand for product $i$ (from column ‘Demand’)
- $I_i$: Initial inventory for product $i$ (from column ‘Initial Inventory’)

**Decision Variables:**  
For each $i \in I$:
- $x_i \in \mathbb{Z}_+$: Number of units of product $i$ to fulfill (non-negative integer)

**Objective:**  
Maximize total revenue:
$$
\max \sum_{i \in I} A_i x_i
$$

**Constraints:**
1. Inventory constraint for each product:
   $$
   x_i \leq I_i \quad \forall i \in I
   $$
2. Demand constraint for each product:
   $$
   x_i \leq d_i \quad \forall i \in I
   $$
3. Non-negativity and integrality:
   $$
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   $$

---

#### Data Mapping

- **Index Set $I$:** All records in table_id `file_0_view_0` where `Product Name` starts with `'S700_'`.
- **Parameter $A_i$:** `Revenue` column in table_id `file_0_view_0`.
- **Parameter $d_i$:** `Demand` column in table_id `file_0_view_0`.
- **Parameter $I_i$:** `Initial Inventory` column in table_id `file_0_view_0`.
- **Decision Variable $x_i$:** Defined for each $i \in I$ as above.