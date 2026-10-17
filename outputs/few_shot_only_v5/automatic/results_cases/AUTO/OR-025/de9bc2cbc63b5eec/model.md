#### Abstract Mathematical Optimization Model

**Index Sets:**
- $I$: Set of all products where the value in column ‘Product Name’ begins with "TABLET_" in /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM16/SmartphoneRetailOutletSalesData.csv.

**Parameters:**
- $A_i$: Revenue per unit of product $i \in I$, from column ‘Revenue’.
- $d_i$: Demand for product $i \in I$, from column ‘Demand’.
- $s_i$: Initial inventory for product $i \in I$, from column ‘Initial Inventory’.

**Decision Variables:**
- $x_i \in \mathbb{Z}_+, \quad \forall i \in I$: Number of units of product $i$ to fulfill (integer, nonnegative).

**Objective:**
\[
\max \sum_{i \in I} A_i x_i
\]

**Constraints:**
1. **Inventory Bound:** 
   \[
   x_i \leq s_i, \quad \forall i \in I
   \]
2. **Demand Bound:** 
   \[
   x_i \leq d_i, \quad \forall i \in I
   \]
3. **Nonnegativity and Integrality:** 
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

**Data Mapping:**

- **Source file:** /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM16/SmartphoneRetailOutletSalesData.csv
- **Index set $I$:** All rows where ‘Product Name’ begins with "TABLET_".
- **Parameter $A_i$:** Column ‘Revenue’.
- **Parameter $d_i$:** Column ‘Demand’.
- **Parameter $s_i$:** Column ‘Initial Inventory’.