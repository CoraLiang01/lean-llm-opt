#### Index Sets

- $I$: Set of all dairy products, indexed by $i$.

#### Parameters

- $A_i$: Revenue per unit for product $i$ (from column "Revenue" in table_id: file_0_view_0).
- $d_i$: Demand for product $i$ (from column "Demand" in table_id: file_0_view_0).
- $I_i$: Initial inventory for product $i$ (from column "Initial Inventory" in table_id: file_0_view_0).

#### Decision Variables

- $x_i$: Number of units of product $i$ to fulfill, $\forall i \in I$.

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

3. **Nonnegativity and Integrality**:
   $$
   x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
   $$

---

#### Data Mapping

- All parameters ($A_i$, $d_i$, $I_i$) and index set $I$ are taken from table_id: file_0_view_0, columns "Full_Product_Name", "Revenue", "Demand", and "Initial Inventory" in the file DairyGoodsSalesDataset.csv. No additional filters were applied; all records returned by CSVQA are included.