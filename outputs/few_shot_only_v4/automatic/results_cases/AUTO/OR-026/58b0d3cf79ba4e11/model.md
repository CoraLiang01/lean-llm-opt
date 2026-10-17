#### Abstract Mathematical Model

**Index Sets:**
- $F$: Set of all products classified as ‘Fashion’ in the dataset.

**Parameters:**
- $a_i$: Revenue per unit of product $i \in F$ (from column ‘Revenue’).
- $d_i$: Deterministic demand for product $i \in F$ (from column ‘Demand’).
- $I_i$: Initial inventory for product $i \in F$ (from column ‘Initial Inventory’).

**Decision Variables:**
- $x_i$: Number of units of product $i \in F$ to fulfill, $x_i \in \mathbb{Z}_+$ (non-negative integers).

**Objective:**
\[
\max \sum_{i \in F} a_i x_i
\]

**Constraints:**
1. **Demand fulfillment:** 
   \[
   x_i \leq d_i \quad \forall i \in F
   \]
2. **Inventory availability:** 
   \[
   x_i \leq I_i \quad \forall i \in F
   \]
3. **Non-negativity and integrality:** 
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in F
   \]

---

#### Data Mapping

- **Source file:** `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM17/SupermarketSales.csv`
- **Index set $F$:** All rows where the ‘Product Name’ column indicates a ‘Fashion’ product (i.e., product names beginning with or containing ‘Fashion’).
- **Parameter $a_i$:** Value from column ‘Revenue’ for each $i \in F$.
- **Parameter $d_i$:** Value from column ‘Demand’ for each $i \in F$.
- **Parameter $I_i$:** Value from column ‘Initial Inventory’ for each $i \in F$.

No other columns or files are used. No additional constraints or subsets are imposed beyond those stated above.