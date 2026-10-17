#### Abstract Mathematical Optimization Model

**Index Sets:**
- $I$: Set of all dairy products (indexed by $i$).

**Parameters:**
- $A_i$: Revenue per unit of product $i$.  
  (From column "Revenue" in /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM22/DairyGoodsSalesDataset.csv)
- $d_i$: Demand for product $i$.  
  (From column "Demand" in the same file)
- $I_i$: Initial inventory for product $i$.  
  (From column "Initial Inventory" in the same file)

**Decision Variables:**
- $x_i$: Number of units of product $i$ to fulfill, for all $i \in I$.

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
3. **Non-negativity:**  
   \[
   x_i \geq 0, \quad \forall i \in I
   \]

**Variable Domain:**  
Unless otherwise specified, $x_i$ are non-negative real variables.

---

#### Data Mapping

- **Index Set $I$:** All rows in /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM22/DairyGoodsSalesDataset.csv, column "Full_Product_Name".
- **Parameter $A_i$:** Column "Revenue" in the same file.
- **Parameter $d_i$:** Column "Demand" in the same file.
- **Parameter $I_i$:** Column "Initial Inventory" in the same file.