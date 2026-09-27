#### Abstract Mathematical Model

**Index Set:**
- $I$ : Set of all products with $id\_number = \text{id999}$.

**Parameters:**
- $A_i$ : Revenue per unit of product $i \in I$ (from column ‘Revenue’ in table_id=file_0_view_0).
- $d_i$ : Demand for product $i \in I$ (from column ‘Demand’ in table_id=file_0_view_0).
- $I_i$ : Initial inventory of product $i \in I$ (from column ‘Initial Inventory’ in table_id=file_0_view_0).

**Decision Variables:**
- $x_i$ : Number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+, \forall i \in I$.

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
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

- Table: OnlineRetailSalesDataset.csv (table_id=file_0_view_0)
    - Product identifier: id_number (filtered to ‘id999’)
    - Revenue per unit: Revenue
    - Demand: Demand
    - Initial Inventory: Initial Inventory