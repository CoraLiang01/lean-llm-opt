#### Abstract Mathematical Model

**Index Set:**
- $I$: Set of “4U” products (from all records in table_id = file_0_view_0).

**Parameters:**
- $A_i$: Revenue per unit of product $i$, from column “Revenue” in table_id = file_0_view_0.
- $d_i$: Demand for product $i$, from column “Demand” in table_id = file_0_view_0.
- $I_i$: Initial inventory of product $i$, from column “Initial Inventory” in table_id = file_0_view_0.

**Decision Variables:**
- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_{\geq 0}$, for all $i \in I$.

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. **Inventory Constraint:** 
   \[
   x_i \leq I_i \quad \forall i \in I
   \]
2. **Demand Constraint:** 
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
3. **Non-negativity and Integrality:**
   \[
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
   \]

---

**Data Mapping:**

- All data for “4U” products is retrieved from table_id = file_0_view_0 (file: OnlineSalesinUSA.csv), columns: “Product Name”, “Revenue”, “Demand”, “Initial Inventory”, with the filter: Product Name starts with “4U”. No further filtering or aggregation is applied. Each parameter is mapped directly from its respective column.