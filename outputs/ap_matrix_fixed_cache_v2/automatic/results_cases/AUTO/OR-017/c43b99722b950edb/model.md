#### Abstract Mathematical Model

**Index Sets:**
- Let $\mathcal{I}$ be the set of all products classified under ‘ZZ’ (indexed by $i$).

**Parameters:**
- $A_i$: Revenue per unit of product $i$ (from column "Revenue" in table_id file_0_view_0).
- $d_i$: Total demand for product $i$ (from column "Demand" in table_id file_0_view_0).
- $I_i$: Initial inventory of product $i$ (from column "Initial Inventory" in table_id file_0_view_0).

**Decision Variables:**
- $x_i$: Number of units of product $i$ to fulfill, for all $i \in \mathcal{I}$.
  - Domain: $x_i \in \mathbb{Z}_+$ (non-negative integers).

**Objective:**
\[
\max \sum_{i \in \mathcal{I}} A_i \cdot x_i
\]

**Constraints:**
1. **Inventory Constraint:** 
   \[
   x_i \leq I_i \quad \forall i \in \mathcal{I}
   \]
2. **Demand Constraint:** 
   \[
   x_i \leq d_i \quad \forall i \in \mathcal{I}
   \]
3. **Non-negativity and Integrality:**
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in \mathcal{I}
   \]

---

#### Data Mapping

- **Index Set $\mathcal{I}$:** All records in table_id file_0_view_0 where column "SKU" has prefix "ZZ".
- **Parameter $A_i$:** Column "Revenue" in table_id file_0_view_0.
- **Parameter $d_i$:** Column "Demand" in table_id file_0_view_0.
- **Parameter $I_i$:** Column "Initial Inventory" in table_id file_0_view_0.
- **Decision Variable $x_i$:** Number of units fulfilled for each $i \in \mathcal{I}$.

All data is sourced from table_id file_0_view_0, with the filter: "SKU" prefix "ZZ".