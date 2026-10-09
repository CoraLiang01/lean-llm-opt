---

**Sets:**  
- $I$ : Set of all dairy products, indexed by $i$.

**Parameters:**  
- $A_i$ : Revenue per unit of product $i$ (from `file_0_view_0`, column `Revenue`).
- $d_i$ : Demand for product $i$ (from `file_0_view_0`, column `Demand`).
- $I_i$ : Initial inventory for product $i$ (from `file_0_view_0`, column `Initial Inventory`).

**Decision Variables:**  
- $x_i$ : Number of units of product $i$ to fulfill, $\forall i \in I$.

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

**Data Mapping:**  
- Table: `file_0_view_0` (from `DairyGoodsSalesDataset.csv`)
    - Product identifier: `Full_Product_Name`
    - Revenue per unit: `Revenue`
    - Demand: `Demand`
    - Initial inventory: `Initial Inventory`