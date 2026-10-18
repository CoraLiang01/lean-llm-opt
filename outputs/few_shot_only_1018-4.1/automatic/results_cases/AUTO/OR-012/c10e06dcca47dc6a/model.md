#### Abstract Mathematical Optimization Model

**Index Sets:**
- $I$: Set of all products (indexed by $i$).

**Parameters:**
- $A_i$: Revenue per unit of product $i$. (From column ‘Revenue’)
- $d_i$: Deterministic demand for product $i$ over the sales horizon. (From column ‘Demand’)
- $I_i$: Initial inventory available for product $i$. (From column ‘Initial Inventory’)

**Decision Variables:**
- $x_i$: Number of units of product $i$ to fulfill for customer purchases.
  - Domain: $x_i \in \mathbb{Z}_+$ (non-negative integers), for all $i \in I$.

**Objective Function:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. **Demand Fulfillment Constraint:**
   \[
   x_i \leq d_i, \quad \forall i \in I
   \]
2. **Inventory Constraint:**
   \[
   x_i \leq I_i, \quad \forall i \in I
   \]
3. **Non-negativity and Integrality:**
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- **Source File:** `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM3/OnlineSalesDataset.csv`
- **Index Set $I$:** All rows in column `Product Name`
- **Parameter $A_i$:** Column `Revenue`
- **Parameter $d_i$:** Column `Demand`
- **Parameter $I_i$:** Column `Initial Inventory`

All rows and columns from the source file are included in the model. No subset or filter is applied.