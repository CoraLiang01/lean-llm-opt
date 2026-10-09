#### Abstract Mathematical Optimization Model

**Index Sets:**

- $I$: Set of all products (across all categories).

**Parameters:**

- $A_i$: Revenue per unit for product $i \in I$ (from column "Revenue").
- $d_i$: Total demand for product $i \in I$ (from column "Demand").
- $I_i$: Initial inventory for product $i \in I$ (from column "Initial Inventory").

**Decision Variables:**

- $x_i$: Number of units of product $i$ to fulfill (integer, $x_i \geq 0$).

**Objective:**

$$
\max \sum_{i \in I} A_i \cdot x_i
$$

**Constraints:**

1. **Inventory Constraints:**
   $$
   x_i \leq I_i \quad \forall i \in I
   $$
2. **Demand Constraints:**
   $$
   x_i \leq d_i \quad \forall i \in I
   $$
3. **Variable Domain:**
   $$
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
   $$

---

#### Data Mapping

- **Source Table:** `file_0_view_0` (from `RetailSalesDataset.csv`)
- **Product Identifier:** "Product Name"
- **Revenue Parameter:** "Revenue"
- **Demand Parameter:** "Demand"
- **Initial Inventory Parameter:** "Initial Inventory"

All parameters are mapped directly from the specified columns in the source table for every product and category.