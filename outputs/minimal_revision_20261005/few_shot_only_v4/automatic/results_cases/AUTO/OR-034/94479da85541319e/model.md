#### Abstract Mathematical Model

**Index Sets:**
- $I$: Set of all baked goods (indexed by $i$).

**Parameters:**
- $A_i$: Revenue per unit of baked good $i$ (from column "Revenue").
- $d_i$: Total demand for baked good $i$ (from column "Demand").
- $I_i$: Initial inventory for baked good $i$ (from column "Initial Inventory").

**Decision Variables:**
- $x_i$: Quantity of baked good $i$ to fulfill, $\forall i \in I$.

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. **Demand fulfillment:**  
   \[
   x_i \leq d_i, \quad \forall i \in I
   \]
2. **Inventory availability:**  
   \[
   x_i \leq I_i, \quad \forall i \in I
   \]
3. **Non-negativity and integrality:**  
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- **Table:** `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM25/Frenchbakerydailysales.csv`
- **Columns:**
  - Product Name $\rightarrow$ Index set $I$
  - Revenue $\rightarrow$ Parameter $A_i$
  - Demand $\rightarrow$ Parameter $d_i$
  - Initial Inventory $\rightarrow$ Parameter $I_i$