#### Sets
- $I$: set of all products (indexed by $i$).

#### Parameters
- $A_i$: revenue per unit of product $i$ (from column "Revenue").
- $d_i$: total demand for product $i$ over the sales horizon (from column "Demand").
- $I_i$: initial inventory of product $i$ (from column "Initial Inventory").

#### Decision Variables
- $x_i$: number of units of product $i$ to fulfill for customer purchases, $\forall i \in I$.

#### Objective
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

#### Constraints
1. **Inventory Constraint:** 
   \[
   x_i \leq I_i, \quad \forall i \in I
   \]
2. **Demand Constraint:** 
   \[
   x_i \leq d_i, \quad \forall i \in I
   \]
3. **Non-negativity and Integrality:**
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- **Set $I$ (products):** All rows in table_id: `file_0_view_0`, column: `"Product Name"`.
- **Parameter $A_i$ (revenue):** table_id: `file_0_view_0`, column: `"Revenue"`.
- **Parameter $d_i$ (demand):** table_id: `file_0_view_0`, column: `"Demand"`.
- **Parameter $I_i$ (initial inventory):** table_id: `file_0_view_0`, column: `"Initial Inventory"`.
- **Variable $x_i$:** defined for all $i \in I$.

All parameters are mapped directly from the corresponding columns in the source table. The model includes all products listed in the data.