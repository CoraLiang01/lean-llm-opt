#### Index Sets

- $I$: set of all products, indexed by $i$.

#### Parameters

- $A_i$: revenue per unit of product $i$ (from column "Revenue" in table_id: file_0_view_0).
- $d_i$: total demand for product $i$ over the sales horizon (from column "Demand" in table_id: file_0_view_0).
- $I_i$: initial inventory of product $i$ (from column "Initial Inventory" in table_id: file_0_view_0).

#### Decision Variables

- $x_i$: number of units of product $i$ to fulfill for customer purchases, $x_i \in \mathbb{Z}_+, \forall i \in I$.

#### Objective

$$
\max \sum_{i \in I} A_i \cdot x_i
$$

#### Constraints

1. **Inventory Constraint**:
   $$
   x_i \leq I_i, \quad \forall i \in I
   $$
2. **Demand Constraint**:
   $$
   x_i \leq d_i, \quad \forall i \in I
   $$
3. **Nonnegativity and Integrality**:
   $$
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   $$

---

#### Data Mapping

- **Source Table**: file_0_view_0 (from OnlineSalesDataset.csv)
- **Product Index Set ($I$)**: All rows in column "Product Name"
- **Revenue ($A_i$)**: column "Revenue"
- **Demand ($d_i$)**: column "Demand"
- **Initial Inventory ($I_i$)**: column "Initial Inventory"