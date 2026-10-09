#### Abstract Mathematical Optimization Model

**Index Set:**
- $I$ : set of all dairy products (indexed by $i$)

**Parameters:**
- $A_i$ : revenue per unit of product $i$ (from column "Revenue")
- $d_i$ : total demand for product $i$ (from column "Demand")
- $I_i$ : initial inventory for product $i$ (from column "Initial Inventory")

**Decision Variables:**
- $x_i$ : number of units of product $i$ to fulfill, $\forall i \in I$

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

#### Data Mapping

- Table: DairyGoodsSalesDataset.csv (table_id: file_0_view_0)
    - Product index: Full_Product_Name
    - Revenue parameter: Revenue
    - Demand parameter: Demand
    - Initial inventory parameter: Initial Inventory