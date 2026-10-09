### Abstract Mathematical Optimization Model

#### Index Sets
- $I$: Set of all products (indexed by $i$).

#### Parameters
- $A_i$: Revenue per unit for product $i$ (from column "Revenue").
- $d_i$: Total demand for product $i$ over the sales horizon (from column "Demand").
- $I_i$: Initial inventory available for product $i$ (from column "Initial Inventory").

#### Decision Variables
- $x_i$: Number of units of product $i$ to fulfill for customer purchases.
  - Domain: $x_i \in \mathbb{Z}_+, \quad \forall i \in I$ (non-negative integers).

#### Objective
$$
\max \sum_{i \in I} A_i \cdot x_i
$$

#### Constraints
1. **Inventory Constraints:**
   $$
   x_i \leq I_i, \quad \forall i \in I
   $$
2. **Demand Constraints:**
   $$
   x_i \leq d_i, \quad \forall i \in I
   $$
3. **Non-negativity and Integrality:**
   $$
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   $$

---

#### Data Mapping

- **Table:** `file_0_view_0` (from `OnlineSalesDataset.csv`)
- **Columns:**
  - Product Name: Index set $I$
  - Revenue: Parameter $A_i$
  - Demand: Parameter $d_i$
  - Initial Inventory: Parameter $I_i$
- **Selection:** All rows (no filter; FALLBACK_FULL_DATA not required)

---

This model maximizes total revenue by choosing integer fulfillment quantities for each product, subject to inventory and demand limits, using all data from the specified columns and table.