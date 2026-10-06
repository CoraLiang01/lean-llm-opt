#### Index Sets

- $I$: Set of all dairy products (indexed by $i$), corresponding to all unique values in column **Full_Product_Name**.

#### Parameters

- $A_i$: Revenue per unit of product $i$ (**Revenue** column, table_id: DairyGoodsSalesDataset.csv).
- $d_i$: Total deterministic demand for product $i$ (**Demand** column, table_id: DairyGoodsSalesDataset.csv).
- $I_i$: Initial inventory available for product $i$ (**Initial Inventory** column, table_id: DairyGoodsSalesDataset.csv).

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

3. **Nonnegativity and Integrality** (fulfilled units are nonnegative integers):
   $$
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   $$

---

### Data Mapping

- **Index Set $I$**: All unique values in column **Full_Product_Name** from table_id: DairyGoodsSalesDataset.csv
- **Parameter $A_i$**: Column **Revenue** from table_id: DairyGoodsSalesDataset.csv
- **Parameter $d_i$**: Column **Demand** from table_id: DairyGoodsSalesDataset.csv
- **Parameter $I_i$**: Column **Initial Inventory** from table_id: DairyGoodsSalesDataset.csv