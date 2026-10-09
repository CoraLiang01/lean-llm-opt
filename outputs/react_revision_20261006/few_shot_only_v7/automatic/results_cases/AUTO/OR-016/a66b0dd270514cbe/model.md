#### Symbolic Model

**Index Sets:**
- $I$: set of all products (indexed by $i$), where each $i$ corresponds to a unique "Product Name".

**Parameters:**
- $A_i$: revenue per unit of product $i$ ("Revenue" column).
- $d_i$: total demand for product $i$ ("Demand" column).
- $I_i$: initial inventory for product $i$ ("Initial Inventory" column).

**Decision Variables:**
- $x_i$: number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$ (non-negative integers), for all $i \in I$.

**Objective:**
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

- Table: `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM7/RetailSalesDataset.csv`
    - Index set $I$: "Product Name"
    - Parameter $A_i$: "Revenue"
    - Parameter $d_i$: "Demand"
    - Parameter $I_i$: "Initial Inventory"