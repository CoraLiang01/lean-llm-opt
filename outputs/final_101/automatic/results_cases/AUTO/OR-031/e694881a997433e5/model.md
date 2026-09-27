#### Abstract Mathematical Model

**Index Sets:**

- $I$ : set of all dairy products (indexed by $i$)

**Parameters:**

- $A_i$ : revenue per unit of product $i$ (from column "Revenue" in table_id: file_0_view_0)
- $d_i$ : total demand for product $i$ (from column "Demand" in table_id: file_0_view_0)
- $I_i$ : initial inventory for product $i$ (from column "Initial Inventory" in table_id: file_0_view_0)

**Decision Variables:**

- $x_i$ : number of units of product $i$ to fulfill, $\forall i \in I$

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

- **Index Set $I$:** All rows in table_id: file_0_view_0 ("DairyGoodsSalesDataset.csv") with column "Full_Product_Name"
- **Parameter $A_i$:** table_id: file_0_view_0, column "Revenue"
- **Parameter $d_i$:** table_id: file_0_view_0, column "Demand"
- **Parameter $I_i$:** table_id: file_0_view_0, column "Initial Inventory"