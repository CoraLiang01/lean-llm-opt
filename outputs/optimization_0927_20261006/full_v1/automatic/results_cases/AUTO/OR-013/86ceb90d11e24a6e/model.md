#### Abstract Mathematical Optimization Model

**Index Set:**

- $I$ : Set of all “4U” products (indexed by $i$).

**Parameters:**

- $A_i$ : Revenue per unit of product $i$ (from column “Revenue” in table_id: file_0_view_0).
- $d_i$ : Demand for product $i$ over the sales horizon (from column “Demand” in table_id: file_0_view_0).
- $I_i$ : Initial inventory of product $i$ (from column “Initial Inventory” in table_id: file_0_view_0).

**Decision Variables:**

- $x_i$ : Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+, \forall i \in I$.

**Objective:**

$$
\max \sum_{i \in I} A_i \cdot x_i
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

- **Index Set $I$:** All records in table_id: file_0_view_0 where “Product Name” starts with “4U”.
- **Parameter $A_i$:** “Revenue” column in table_id: file_0_view_0.
- **Parameter $d_i$:** “Demand” column in table_id: file_0_view_0.
- **Parameter $I_i$:** “Initial Inventory” column in table_id: file_0_view_0.