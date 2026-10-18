#### Abstract Mathematical Optimization Model

**Index Sets:**
- $P$: Set of all products (indexed by $i$).

**Parameters:**
- $A_i$: Revenue per unit for product $i$ (from column ‘Revenue’ in /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM7/RetailSalesDataset.csv).
- $d_i$: Demand for product $i$ (from column ‘Demand’).
- $I_i$: Initial inventory for product $i$ (from column ‘Initial Inventory’).

**Decision Variables:**
- $x_i$: Quantity of product $i$ to fulfill, $x_i \geq 0$, $\forall i \in P$.

**Objective:**
\[
\max \sum_{i \in P} A_i \cdot x_i
\]

**Constraints:**
1. **Demand Constraint:** 
   \[
   x_i \leq d_i, \quad \forall i \in P
   \]
2. **Inventory Constraint:** 
   \[
   x_i \leq I_i, \quad \forall i \in P
   \]
3. **Non-negativity:**
   \[
   x_i \geq 0, \quad \forall i \in P
   \]

---

#### Data Mapping

- **Source file:** /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM7/RetailSalesDataset.csv
- **Index set $P$:** All rows in column ‘Product Name’.
- **Parameter $A_i$:** Column ‘Revenue’.
- **Parameter $d_i$:** Column ‘Demand’.
- **Parameter $I_i$:** Column ‘Initial Inventory’.