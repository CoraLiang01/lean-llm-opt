---

### Abstract Mathematical Model

#### Index Sets
- $I$: Set of all products in the dataset.

#### Parameters
- $A_i$: Revenue per unit for product $i \in I$ (from column "Revenue").
- $d_i$: Deterministic demand for product $i \in I$ (from column "Demand").
- $I_i$: Initial inventory for product $i \in I$ (from column "Initial Inventory").

#### Decision Variables
- $x_i$: Number of units of product $i$ to fulfill for customer purchases.
  - Domain: $x_i \in \mathbb{Z}_+$ (non-negative integers), $\forall i \in I$

#### Objective
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

#### Constraints
1. **Inventory and Demand Fulfillment Bounds:**
   \[
   0 \leq x_i \leq \min\{d_i, I_i\}, \quad \forall i \in I
   \]

---

### Data Mapping

- **Source Table:** `file_0_view_0` (from `OnlineSalesDataset.csv`)
- **Index Set $I$:** All records in column `"Product Name"`
- **Parameter $A_i$:** Column `"Revenue"` in `file_0_view_0`
- **Parameter $d_i$:** Column `"Demand"` in `file_0_view_0`
- **Parameter $I_i$:** Column `"Initial Inventory"` in `file_0_view_0`
- **Selection:** All records in the table are included (no filters applied).

---