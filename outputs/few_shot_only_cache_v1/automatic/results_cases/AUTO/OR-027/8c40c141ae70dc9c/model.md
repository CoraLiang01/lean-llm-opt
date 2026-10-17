#### Abstract Optimization Model

**Index Sets:**
- $I$ : Set of all products in table_id = file_0_view_0 (SupermartGrocerySales-RetailAnalyticsDataset.csv) such that the value in column "Sub Category" contains the substring "Organ" (i.e., all products classified as ‘Organ’).

**Parameters:**
- $A_i$ : Revenue per unit of product $i \in I$, from column "Revenue".
- $d_i$ : Demand for product $i \in I$, from column "Demand".
- $I_i$ : Initial inventory for product $i \in I$, from column "Initial Inventory".

**Decision Variables:**
- $x_i$ : Number of units of product $i \in I$ to fulfill; $x_i \in \mathbb{Z}_+$ (nonnegative integer).

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. **Demand and Inventory Bounds:**
   \[
   0 \leq x_i \leq \min\{d_i, I_i\} \quad \forall i \in I
   \]
   (Equivalently, $x_i \leq d_i$ and $x_i \leq I_i$ for all $i \in I$.)

2. **Integrality:**
   \[
   x_i \in \mathbb{Z}_+ \quad \forall i \in I
   \]

---

#### Data Mapping

- **Source Table:** table_id = file_0_view_0 (SupermartGrocerySales-RetailAnalyticsDataset.csv)
- **Index Set $I$:** All rows where "Sub Category" contains the substring "Organ".
- **Parameter $A_i$:** Column "Revenue", for each $i \in I$.
- **Parameter $d_i$:** Column "Demand", for each $i \in I$.
- **Parameter $I_i$:** Column "Initial Inventory", for each $i \in I$.
- **Decision Variable $x_i$:** Defined for each $i \in I$.

No other columns or rows are used. No additional constraints or selection rules are imposed beyond those stated above.