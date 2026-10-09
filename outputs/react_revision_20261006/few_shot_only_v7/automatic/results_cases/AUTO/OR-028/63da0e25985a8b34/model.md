#### Symbolic Model

**Index Sets:**
- $I$: set of products, indexed by $i$

**Parameters:**
- $A_i$: revenue per unit of product $i$ (from column "Revenue")
- $d_i$: total demand for product $i$ (from column "Demand")
- $I_i$: initial inventory for product $i$ (from column "Initial Inventory")

**Decision Variables:**
- $x_i$: number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. **Demand fulfillment cannot exceed demand:**
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
2. **Demand fulfillment cannot exceed initial inventory:**
   \[
   x_i \leq I_i \quad \forall i \in I
   \]
3. **Non-negativity and integrality:**
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

#### Data Mapping

- Table: `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM19/WomenClothingEcommerceSalesData.csv`
    - Product index set $I$: from column "Product Name"
    - Revenue parameter $A_i$: from column "Revenue"
    - Demand parameter $d_i$: from column "Demand"
    - Initial inventory parameter $I_i$: from column "Initial Inventory"