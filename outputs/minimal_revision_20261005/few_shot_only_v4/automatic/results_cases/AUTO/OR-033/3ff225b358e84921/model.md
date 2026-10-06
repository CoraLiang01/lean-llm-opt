#### Abstract Mathematical Model

**Index Sets:**

- $I$ : Set of all products classified as ‘Baby’.

**Parameters:**

- $A_i$ : Revenue per unit of product $i \in I$.
- $d_i$ : Total deterministic demand for product $i \in I$.
- $I_i$ : Initial inventory available for product $i \in I$.

**Decision Variables:**

- $x_i$ : Number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+$.

**Objective:**

\[
\max \quad \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**

1. **Inventory Constraint:**  
   \[
   x_i \leq I_i, \quad \forall i \in I
   \]

2. **Demand Constraint:**  
   \[
   x_i \leq d_i, \quad \forall i \in I
   \]

3. **Non-negativity and Integrality:**  
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- **Source Table:** `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM24/EuropeSalesRecords.csv`
- **Index Set $I$:** All records where `Product Name` indicates a ‘Baby’ product.
- **Parameter $A_i$:** `Revenue` column.
- **Parameter $d_i$:** `Demand` column.
- **Parameter $I_i$:** `Initial Inventory` column.