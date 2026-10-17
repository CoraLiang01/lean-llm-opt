#### Abstract Mathematical Optimization Model

**Index Sets:**
- $I$: Set of all baked goods in the dataset (indexed by $i$).

**Parameters:**
- $A_i$: Revenue per unit of baked good $i$ (from column ‘Revenue’).
- $d_i$: Total demand for baked good $i$ over the sales horizon (from column ‘Demand’).
- $I_i$: Initial inventory of baked good $i$ (from column ‘Initial Inventory’).

**Decision Variables:**
- $x_i$: Quantity of baked good $i$ to fulfill (integer, $x_i \geq 0$).

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. **Demand fulfillment:**
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
2. **Inventory limit:**
   \[
   x_i \leq I_i \quad \forall i \in I
   \]
3. **Non-negativity and integrality:**
   \[
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
   \]

---

#### Data Mapping

- **File:** /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM25/Frenchbakerydailysales.csv
- **Index Set $I$:** All rows in column ‘Product Name’
- **Parameter $A_i$:** Column ‘Revenue’
- **Parameter $d_i$:** Column ‘Demand’
- **Parameter $I_i$:** Column ‘Initial Inventory’