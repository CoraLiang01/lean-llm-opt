#### Abstract Mathematical Model

**Index Set:**
- $I$ : set of all baked goods, indexed by $i$

**Parameters:**
- $A_i$ : revenue per unit of baked good $i$ (from column "Revenue")
- $d_i$ : total demand for baked good $i$ (from column "Demand")
- $I_i$ : initial inventory for baked good $i$ (from column "Initial Inventory")

**Decision Variables:**
- $x_i$ : quantity of baked good $i$ to fulfill, $x_i \in \mathbb{Z}_+, \forall i \in I$

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. **Demand fulfillment:** 
   \[
   x_i \leq d_i, \quad \forall i \in I
   \]
2. **Inventory limit:** 
   \[
   x_i \leq I_i, \quad \forall i \in I
   \]
3. **Non-negativity and integrality:** 
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- Table: `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM25/Frenchbakerydailysales.csv`
    - Index set $I$: all unique values in column `"Product Name"`
    - Parameter $A_i$: column `"Revenue"`
    - Parameter $d_i$: column `"Demand"`
    - Parameter $I_i$: column `"Initial Inventory"`