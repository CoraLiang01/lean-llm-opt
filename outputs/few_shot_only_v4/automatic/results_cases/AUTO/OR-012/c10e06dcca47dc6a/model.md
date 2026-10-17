#### Abstract Mathematical Optimization Model

**Index Sets:**
- $I$: Set of all products, indexed by $i$.

**Parameters:**
- $A_i$: Revenue per unit for product $i$. (from column ‘Revenue’)
- $d_i$: Deterministic demand for product $i$ over the sales horizon. (from column ‘Demand’)
- $I_i$: Initial inventory available for product $i$. (from column ‘Initial Inventory’)

**Decision Variables:**
- $x_i$: Number of units of product $i$ to fulfill for customer purchases.
  - Domain: $x_i \in \mathbb{Z}_+$ (non-negative integers), for all $i \in I$.

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

- **File:** `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM3/OnlineSalesDataset.csv`
- **Index Set $I$:** All rows in column `Product Name`
- **Parameter $A_i$:** Column `Revenue`
- **Parameter $d_i$:** Column `Demand`
- **Parameter $I_i$:** Column `Initial Inventory`