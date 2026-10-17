#### Abstract Mathematical Optimization Model

**Index Set:**

- $I$: Set of all products classified under ‘ELE-S’ (i.e., all products whose identifier in the ‘Product_Reference’ column begins with "ELE-S").

**Parameters:**

- $a_i$: Revenue per unit of product $i \in I$ (from column ‘Revenue’ in /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM14/SalesStoreoverview.csv).
- $d_i$: Deterministic demand for product $i \in I$ (from column ‘Demand’).
- $s_i$: Initial inventory for product $i \in I$ (from column ‘Initial Inventory’).

**Decision Variables:**

- $x_i$: Number of units of product $i \in I$ to fulfill, $x_i \geq 0$ (continuous, nonnegative).

**Objective:**

\[
\max \sum_{i \in I} a_i x_i
\]

**Constraints:**

1. **Demand fulfillment constraint:**
   \[
   x_i \leq d_i \qquad \forall i \in I
   \]

2. **Inventory constraint:**
   \[
   x_i \leq s_i \qquad \forall i \in I
   \]

3. **Nonnegativity:**
   \[
   x_i \geq 0 \qquad \forall i \in I
   \]

---

#### Data Mapping

- **File:** /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM14/SalesStoreoverview.csv
- **Index Set $I$:** All rows where ‘Product_Reference’ begins with "ELE-S".
- **Parameter $a_i$:** Column ‘Revenue’.
- **Parameter $d_i$:** Column ‘Demand’.
- **Parameter $s_i$:** Column ‘Initial Inventory’.