#### Abstract Mathematical Model

**Index Set:**
- $I$ : Set of all products where `Product Name` begins with "TABLET" (from the current table).

**Parameters:**
- $A_i$ : Revenue per unit of product $i \in I$ (from column `Revenue`).
- $d_i$ : Demand for product $i \in I$ (from column `Demand`).
- $s_i$ : Initial inventory for product $i \in I$ (from column `Initial Inventory`).

**Decision Variables:**
- $x_i$ : Number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+$ (non-negative integers).

**Objective:**
\[
\max \sum_{i \in I} A_i x_i
\]

**Constraints:**
1. **Inventory Constraint:**  
   \[
   x_i \leq s_i \quad \forall i \in I
   \]
2. **Demand Constraint:**  
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
3. **Non-negativity and Integrality:**  
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- **Source Table:** `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM16/SmartphoneRetailOutletSalesData.csv`
- **Table ID:** `file_0_view_0`
- **Index Set $I$:** All records where `Product Name` starts with "TABLET"
- **Parameter $A_i$:** Column `Revenue`
- **Parameter $d_i$:** Column `Demand`
- **Parameter $s_i$:** Column `Initial Inventory`