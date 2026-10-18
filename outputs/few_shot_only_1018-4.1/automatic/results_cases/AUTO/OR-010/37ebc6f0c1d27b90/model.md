#### Abstract Mathematical Optimization Model

**Index Sets:**
- $I$: Set of all products (indexed by $i$).

**Parameters:**
- $A_i$: Revenue per unit for product $i$.  
  (From column "Revenue" in /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM1/MobileSalesDataset.csv)
- $d_i$: Deterministic total demand for product $i$ over the sales cycle.  
  (From column "Demand" in /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM1/MobileSalesDataset.csv)
- $I_i$: Initial inventory available for product $i$ at the start of the sales cycle.  
  (From column "Initial Inventory" in /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM1/MobileSalesDataset.csv)

**Decision Variables:**
- $x_i$: Number of customer orders fulfilled for product $i$; $x_i \in \mathbb{Z}_+, \forall i \in I$.

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

- **File:** /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM1/MobileSalesDataset.csv
    - **Product Index ($I$):** All rows in column "Product Name"
    - **Revenue ($A_i$):** Column "Revenue"
    - **Demand ($d_i$):** Column "Demand"
    - **Initial Inventory ($I_i$):** Column "Initial Inventory"

All rows and columns are included as per the query and CSVQA protocol. No selection or filtering is applied.