#### Abstract Mathematical Optimization Model

**Index Sets:**
- $I$: Set of all products (indexed by $i$).

**Parameters:**
- $A_i$: Revenue per unit of product $i$ (from column ‘Revenue’).
- $d_i$: Total demand for product $i$ over the sales horizon (from column ‘Demand’).
- $I_i$: Initial inventory of product $i$ (from column ‘Initial Inventory’).

**Decision Variables:**
- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+, \forall i \in I$.

**Objective:**
\[
\max \sum_{i \in I} A_i x_i
\]

**Constraints:**
1. **Demand fulfillment cannot exceed demand:**
   \[
   x_i \leq d_i, \quad \forall i \in I
   \]
2. **Demand fulfillment cannot exceed initial inventory:**
   \[
   x_i \leq I_i, \quad \forall i \in I
   \]
3. **Nonnegativity and integrality:**
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- **Source file:** /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM19/WomenClothingEcommerceSalesData.csv
- **Index set $I$:** All rows in column ‘Product Name’
- **Parameter $A_i$:** Column ‘Revenue’
- **Parameter $d_i$:** Column ‘Demand’
- **Parameter $I_i$:** Column ‘Initial Inventory’