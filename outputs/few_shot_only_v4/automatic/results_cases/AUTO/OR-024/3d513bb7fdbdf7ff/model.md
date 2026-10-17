#### Abstract Mathematical Optimization Model

**Index Set:**

- Let $\mathcal{I}$ be the set of all products $i$ such that the value in the column “Product Name” of the file `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM15/SampleSalesData.csv` begins with the prefix `S700_`.

**Parameters:**

- For each $i \in \mathcal{I}$:
    - $A_i$: Revenue per unit of product $i$ (from column “Revenue”)
    - $d_i$: Demand for product $i$ (from column “Demand”)
    - $I_i$: Initial inventory for product $i$ (from column “Initial Inventory”)

**Decision Variables:**

- For each $i \in \mathcal{I}$:
    - $x_i \in \mathbb{Z}_+$: Number of units of product $i$ to fulfill (non-negative integer)

**Objective:**

\[
\max \sum_{i \in \mathcal{I}} A_i \cdot x_i
\]

**Constraints:**

1. **Inventory and Demand Bounds:**
   \[
   0 \leq x_i \leq \min\{I_i,\, d_i\} \qquad \forall i \in \mathcal{I}
   \]

   (Equivalently, two constraints per $i$:)
   \[
   x_i \leq I_i \qquad \forall i \in \mathcal{I}
   \]
   \[
   x_i \leq d_i \qquad \forall i \in \mathcal{I}
   \]
   \[
   x_i \geq 0 \qquad \forall i \in \mathcal{I}
   \]
   \[
   x_i \in \mathbb{Z} \qquad \forall i \in \mathcal{I}
   \]

**Data Mapping:**

- **Source file:** `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM15/SampleSalesData.csv`
- **Index set selection:** $\mathcal{I}$ consists of all rows where the value in the column “Product Name” begins with the prefix `S700_`.
- **Parameter mapping:**
    - $A_i$ is the value in column “Revenue” for product $i$.
    - $d_i$ is the value in column “Demand” for product $i$.
    - $I_i$ is the value in column “Initial Inventory” for product $i$.
- **No other columns or files are used.**