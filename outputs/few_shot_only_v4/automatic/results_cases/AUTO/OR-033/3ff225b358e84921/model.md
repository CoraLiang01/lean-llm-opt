#### Abstract Mathematical Optimization Model

**Index Sets:**

- $I$ : Set of all products classified under ‘Baby’.

**Parameters:**

- $A_i$ : Revenue per unit for product $i \in I$ (from column ‘Revenue’).
- $d_i$ : Deterministic demand for product $i \in I$ (from column ‘Demand’).
- $I_i$ : Initial inventory for product $i \in I$ (from column ‘Initial Inventory’).

**Decision Variables:**

- $x_i$ : Number of units of product $i \in I$ to fulfill, integer, $x_i \geq 0$.

**Objective Function:**

\[
\max \quad \sum_{i \in I} A_i \cdot x_i
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

3. **Variable Domain:**
   \[
   x_i \in \mathbb{Z}_{+} \quad \forall i \in I
   \]

---

#### Data Mapping

- **File:** /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM24/EuropeSalesRecords.csv
- **Index Set Construction:** $I$ is the set of all rows where the ‘Product Name’ column indicates classification under ‘Baby’.
- **Parameters:**
    - $A_i$ : Value from column ‘Revenue’ for product $i$.
    - $d_i$ : Value from column ‘Demand’ for product $i$.
    - $I_i$ : Value from column ‘Initial Inventory’ for product $i$.
- **Decision Variables:** $x_i$ defined for each $i \in I$.

No additional constraints or selection rules are imposed beyond those stated above.