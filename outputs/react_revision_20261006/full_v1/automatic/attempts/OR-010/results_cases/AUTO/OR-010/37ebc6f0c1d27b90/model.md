#### Symbolic Mathematical Model

**Index Set:**

- $I$ : set of all products (indexed by $i$), as defined by all "Product Name" entries in table_id = file_0_view_0.

**Parameters:**

- $A_i$ : revenue per unit for product $i$ (from column "Revenue", table_id = file_0_view_0)
- $d_i$ : total demand for product $i$ during the sales cycle (from column "Demand", table_id = file_0_view_0)
- $I_i$ : initial inventory for product $i$ (from column "Initial Inventory", table_id = file_0_view_0)

**Decision Variables:**

- $x_i$ : number of orders fulfilled for product $i$; $x_i \in \mathbb{Z}_+, \forall i \in I$

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

- **Index Set $I$**: All records in table_id = file_0_view_0, column "Product Name"
- **Parameter $A_i$**: table_id = file_0_view_0, column "Revenue"
- **Parameter $d_i$**: table_id = file_0_view_0, column "Demand"
- **Parameter $I_i$**: table_id = file_0_view_0, column "Initial Inventory"