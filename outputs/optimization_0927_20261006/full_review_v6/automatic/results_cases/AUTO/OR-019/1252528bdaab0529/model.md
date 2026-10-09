#### Abstract Mathematical Optimization Model

**Index Set:**
- $I$: Set of all products classified as ‘27in’.

**Parameters:**
- $A_i$: Revenue per unit for product $i \in I$ (from column ‘Revenue’).
- $d_i$: Demand for product $i \in I$ (from column ‘Demand’).
- $I_i$: Initial inventory for product $i \in I$ (from column ‘Initial Inventory’).

**Decision Variables:**
- $x_i$: Number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+$.

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
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
3. **Non-negativity and Integrality:**
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- **Source Table:** `SalesDataAnalysis.csv`
- **Index Set $I$:** All records where `Product Name` has prefix ‘27in’ (as filtered in the query).
- **Parameter $A_i$:** Column `Revenue` in `SalesDataAnalysis.csv` for each $i \in I$.
- **Parameter $d_i$:** Column `Demand` in `SalesDataAnalysis.csv` for each $i \in I$.
- **Parameter $I_i$:** Column `Initial Inventory` in `SalesDataAnalysis.csv` for each $i \in I$.