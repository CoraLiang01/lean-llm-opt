#### Abstract Mathematical Model

**Index Sets:**
- $I$ : Set of dairy products (indexed by $i$)

**Parameters:**
- $A_i$ : Revenue per unit of product $i$ (from column "Revenue")
- $d_i$ : Demand for product $i$ (from column "Demand")
- $I_i$ : Initial inventory for product $i$ (from column "Initial Inventory")

**Decision Variables:**
- $x_i$ : Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+, \forall i \in I$

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. Inventory constraints:
   \[
   x_i \leq I_i, \quad \forall i \in I
   \]
2. Demand constraints:
   \[
   x_i \leq d_i, \quad \forall i \in I
   \]
3. Non-negativity and integrality:
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- Source Table: DairyGoodsSalesDataset.csv
- Table ID: file_0_view_0
- Columns used:
    - Product identifier: "Full_Product_Name"
    - Revenue per unit: "Revenue"
    - Demand: "Demand"
    - Initial inventory: "Initial Inventory"
- All records in the table are included (no filters applied).