#### Abstract Mathematical Model

**Index Set:**
- $I$: Set of all car models classified under ‘FDK57’ (from the column “Product Name”).

**Parameters:**
- $A_i$: Revenue per unit of car model $i \in I$ (from column “Revenue”).
- $d_i$: Deterministic total demand for car model $i \in I$ (from column “Demand”).
- $I_i$: Initial inventory for car model $i \in I$ (from column “Initial Inventory”).

**Decision Variables:**
- $x_i$: Number of units of car model $i \in I$ to fulfill, integer, $x_i \geq 0$.

**Objective:**
\[
\max \sum_{i \in I} A_i x_i
\]

**Constraints:**
1. **Inventory Constraint:**  
   $\forall i \in I: \quad x_i \leq I_i$
2. **Demand Constraint:**  
   $\forall i \in I: \quad x_i \leq d_i$
3. **Non-negativity and Integrality:**  
   $\forall i \in I: \quad x_i \in \mathbb{Z}_+, \ x_i \geq 0$

---

#### Data Mapping

- **Source Table:** `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv`
- **Columns Used:**
  - `Product Name` (for index set $I$)
  - `Revenue` (for parameter $A_i$)
  - `Demand` (for parameter $d_i$)
  - `Initial Inventory` (for parameter $I_i$)