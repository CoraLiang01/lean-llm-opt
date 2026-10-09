#### Abstract Mathematical Model

**Index Set:**
- $I$ : set of all products with $\texttt{id999}$ in column "id_number" of table_id file_0_view_0.

**Parameters (for each $i \in I$):**
- $A_i$ : revenue per unit of product $i$ (column "Revenue", table_id file_0_view_0)
- $d_i$ : demand for product $i$ over the sales horizon (column "Demand", table_id file_0_view_0)
- $I_i$ : initial inventory of product $i$ (column "Initial Inventory", table_id file_0_view_0)

**Decision Variables:**
- $x_i$ : number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+, \forall i \in I$

**Objective:**
\[
\max \sum_{i \in I} A_i x_i
\]

**Constraints:**
1. Inventory constraint:
   \[
   x_i \leq I_i, \quad \forall i \in I
   \]
2. Demand constraint:
   \[
   x_i \leq d_i, \quad \forall i \in I
   \]
3. Non-negativity and integrality:
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

**Data Mapping:**

- Table: OnlineRetailSalesDataset.csv, table_id file_0_view_0
    - Index set $I$: all rows where "id_number" contains 'id999'
    - Parameter $A_i$: column "Revenue"
    - Parameter $d_i$: column "Demand"
    - Parameter $I_i$: column "Initial Inventory"