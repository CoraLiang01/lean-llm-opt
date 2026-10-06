#### Index Sets

- $I$: Set of all products, indexed by $i$.

#### Parameters

- $A_i$: Revenue per unit of product $i$ (from column "Revenue").
- $d_i$: Demand for product $i$ (from column "Demand").
- $I_i$: Initial inventory for product $i$ (from column "Initial Inventory").

#### Decision Variables

- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$ (non-negative integers), for all $i \in I$.

#### Objective

\[
\max \sum_{i \in I} A_i \cdot x_i
\]

#### Constraints

1. **Demand fulfillment cannot exceed demand:**
   \[
   x_i \leq d_i, \quad \forall i \in I
   \]

2. **Demand fulfillment cannot exceed initial inventory:**
   \[
   x_i \leq I_i, \quad \forall i \in I
   \]

3. **Non-negativity and integrality:**
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- Table: `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM19/WomenClothingEcommerceSalesData.csv`
    - Product Name: index set $I$
    - Revenue: parameter $A_i$
    - Demand: parameter $d_i$
    - Initial Inventory: parameter $I_i$