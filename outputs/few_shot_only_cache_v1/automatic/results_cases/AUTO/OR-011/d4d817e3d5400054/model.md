#### Abstract Mathematical Model

**Index Sets:**
- $I$ : Set of all products $i$ such that $\texttt{id\_number}_i = \text{id999}$ in table file_0_view_0.

**Parameters:**
- $a_i$ : Revenue per unit for product $i$ (from column "Revenue" in file_0_view_0).
- $d_i$ : Total deterministic demand for product $i$ over the sales horizon (from column "Demand" in file_0_view_0).
- $s_i$ : Initial inventory for product $i$ (from column "Initial Inventory" in file_0_view_0).

**Decision Variables:**
- $x_i$ : Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$ (non-negative integer), for all $i \in I$.

**Objective:**
\[
\max \sum_{i \in I} a_i x_i
\]

**Constraints:**
1. **Inventory Constraint:**  
   \[
   x_i \leq s_i \qquad \forall i \in I
   \]
2. **Demand Constraint:**  
   \[
   x_i \leq d_i \qquad \forall i \in I
   \]
3. **Non-negativity and Integrality:**  
   \[
   x_i \in \mathbb{Z}_+, \qquad \forall i \in I
   \]

---

#### Data Mapping

- **Source Table:** file_0_view_0 (from OnlineRetailSalesDataset.csv)
- **Index Set $I$:** All rows where column "id_number" equals "id999".
- **Parameter $a_i$:** Column "Revenue" for each $i \in I$.
- **Parameter $d_i$:** Column "Demand" for each $i \in I$.
- **Parameter $s_i$:** Column "Initial Inventory" for each $i \in I$.
- **Decision Variable $x_i$:** Defined for each $i \in I$.

---

**Note:**  
- Only products with $\texttt{id\_number} = \text{id999}$ are included in the index set $I$.
- All constraints and the objective are applied only to this subset.  
- All data columns referenced are from file_0_view_0, with column names as given.