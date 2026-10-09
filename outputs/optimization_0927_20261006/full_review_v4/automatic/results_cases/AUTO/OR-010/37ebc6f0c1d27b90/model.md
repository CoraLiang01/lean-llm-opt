#### Index Sets

- $I$: set of all products, with each product $i \in I$ identified by its "Product Name" in table_id file_0_view_0.

#### Parameters

- $A_i$: revenue per unit for product $i$, from column "Revenue" in table_id file_0_view_0.
- $d_i$: expected demand for product $i$ during the sales cycle, from column "Demand" in table_id file_0_view_0.
- $I_i$: initial inventory for product $i$, from column "Initial Inventory" in table_id file_0_view_0.

#### Decision Variables

- $x_i$: number of orders fulfilled for product $i$, $x_i \in \mathbb{Z}_+$ (non-negative integer), for all $i \in I$.

#### Objective

$$
\max \sum_{i \in I} A_i \cdot x_i
$$

#### Constraints

1. **Inventory Constraint** (cannot fulfill more than available inventory):
   $$
   x_i \leq I_i, \quad \forall i \in I
   $$

2. **Demand Constraint** (cannot fulfill more than demand):
   $$
   x_i \leq d_i, \quad \forall i \in I
   $$

3. **Non-negativity and Integrality**:
   $$
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   $$

---

#### Data Mapping

- All parameters and index sets are defined using columns "Product Name", "Revenue", "Demand", and "Initial Inventory" from table_id file_0_view_0 in MobileSalesDataset.csv.
- No additional filters were applied; all 71 product records are included as returned by CSVQA.