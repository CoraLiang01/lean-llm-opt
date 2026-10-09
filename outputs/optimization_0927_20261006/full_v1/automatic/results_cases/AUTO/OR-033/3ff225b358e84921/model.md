#### Abstract Mathematical Optimization Model

**Index Sets:**

- $I$ : Set of all products in the 'Baby' category (from the relevant table).

**Parameters:**

- $A_i$ : Revenue per unit of product $i \in I$ (from column 'Revenue', table_id: file_0_view_0).
- $d_i$ : Total deterministic demand for product $i \in I$ (from column 'Demand', table_id: file_0_view_0).
- $I_i$ : Initial inventory for product $i \in I$ (from column 'Initial Inventory', table_id: file_0_view_0).

**Decision Variables:**

- $x_i$ : Number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+$.

**Objective:**

$$
\max \quad \sum_{i \in I} A_i \cdot x_i
$$

**Constraints:**

1. **Inventory Constraint:**
   $$
   x_i \leq I_i, \quad \forall i \in I
   $$

2. **Demand Constraint:**
   $$
   x_i \leq d_i, \quad \forall i \in I
   $$

3. **Non-negativity and Integrality:**
   $$
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   $$

---

#### Data Mapping

- **Table:** EuropeSalesRecords.csv
- **table_id:** file_0_view_0
- **Columns Used:**
  - 'Product Name' (for index set $I$; filtered to 'Baby' products)
  - 'Revenue' (parameter $A_i$)
  - 'Demand' (parameter $d_i$)
  - 'Initial Inventory' (parameter $I_i$)