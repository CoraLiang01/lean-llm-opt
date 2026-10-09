#### Abstract Mathematical Model

**Index Set:**
- $I$: Set of all “4U” products (indexed by $i$).

**Parameters:**
- $A_i$: Revenue per unit of product $i$ (from column “Revenue”).
- $d_i$: Demand for product $i$ over the sales horizon (from column “Demand”).
- $I_i$: Initial inventory of product $i$ (from column “Initial Inventory”).

**Decision Variables:**
- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+, \forall i \in I$.

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
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

- **Table:** file_0_view_0 (from OnlineSalesinUSA.csv)
- **Columns:**
  - Product Name: Used to select “4U” products (prefix match).
  - Revenue: Parameter $A_i$.
  - Demand: Parameter $d_i$.
  - Initial Inventory: Parameter $I_i$.
- **Filter Applied:** Only rows where “Product Name” starts with “4U” (prefix match).
- **Returned Records:** All “4U” products in the table, with their “Revenue”, “Demand”, and “Initial Inventory” values.