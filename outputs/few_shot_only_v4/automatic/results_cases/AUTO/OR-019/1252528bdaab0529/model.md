#### Abstract Mathematical Optimization Model

**Index Set:**

- $I$: Set of all products classified as ‘27in’ (i.e., all products whose name contains ‘27in’ in the Product Name column of /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM10/SalesDataAnalysis.csv).

**Parameters:**

- $A_i$: Revenue per unit of product $i \in I$ (from the ‘Revenue’ column).
- $d_i$: Demand for product $i \in I$ (from the ‘Demand’ column).
- $I_i$: Initial inventory for product $i \in I$ (from the ‘Initial Inventory’ column).

**Decision Variables:**

- $x_i$: Number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+$ (non-negative integers).

**Objective Function:**

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
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- **File:** /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM10/SalesDataAnalysis.csv
- **Index Set $I$:** All rows where ‘Product Name’ contains ‘27in’.
- **Parameter $A_i$:** Column ‘Revenue’.
- **Parameter $d_i$:** Column ‘Demand’.
- **Parameter $I_i$:** Column ‘Initial Inventory’.