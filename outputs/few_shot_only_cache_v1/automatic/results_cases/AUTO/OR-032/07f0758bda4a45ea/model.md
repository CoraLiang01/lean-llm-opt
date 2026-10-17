#### Abstract Optimization Model

**Index Sets:**
- $I$ : Set of all products $i$ such that the product is classified as ‘Books’ (i.e., $i \in I$ if and only if the value in column ‘Product_Name’ of table file_0_view_0 begins with ‘Books’).

**Parameters:**
- $a_i$ : Revenue per unit of product $i$ (from column ‘Revenue’ in file_0_view_0).
- $d_i$ : Demand for product $i$ (from column ‘Demand’ in file_0_view_0).
- $s_i$ : Initial inventory for product $i$ (from column ‘Initial Inventory’ in file_0_view_0).

**Decision Variables:**
- $x_i$ : Number of units of product $i$ to fulfill, $\forall i \in I$.

**Objective:**
\[
\max \sum_{i \in I} a_i x_i
\]

**Constraints:**
1. **Demand fulfillment:** 
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
2. **Inventory limit:** 
   \[
   x_i \leq s_i \quad \forall i \in I
   \]
3. **Nonnegativity and integrality:** 
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- **Source Table:** file_0_view_0 (from DifferentStoreSales.csv)
- **Index Set $I$:** All rows where the value in column ‘Product_Name’ begins with ‘Books’.
- **Parameter $a_i$:** Column ‘Revenue’ in file_0_view_0, for $i \in I$.
- **Parameter $d_i$:** Column ‘Demand’ in file_0_view_0, for $i \in I$.
- **Parameter $s_i$:** Column ‘Initial Inventory’ in file_0_view_0, for $i \in I$.
- **Decision Variable $x_i$:** Defined for each $i \in I$.

No other columns or rows are used. No additional constraints or selection rules are imposed beyond those stated above.