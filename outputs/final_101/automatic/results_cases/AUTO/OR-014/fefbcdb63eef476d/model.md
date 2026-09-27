### Abstract Mathematical Optimization Model

#### Index Sets
- $I$: set of all pizza types (from column "Product Name" in table_id file_0_view_0).

#### Parameters
- $A_i$: revenue per unit of pizza type $i$ (from column "Revenue" in table_id file_0_view_0), $\forall i \in I$.
- $d_i$: total demand for pizza type $i$ (from column "Demand" in table_id file_0_view_0), $\forall i \in I$.
- $I_i$: initial inventory available for pizza type $i$ (from column "Initial Inventory" in table_id file_0_view_0), $\forall i \in I$.

#### Decision Variables
- $x_i$: number of units of pizza type $i$ to fulfill, $x_i \in \mathbb{Z}_+$ (non-negative integer), $\forall i \in I$.

#### Objective
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

#### Constraints
1. **Inventory Constraints:**  
   \[
   x_i \leq I_i, \quad \forall i \in I
   \]
2. **Demand Constraints:**  
   \[
   x_i \leq d_i, \quad \forall i \in I
   \]
3. **Non-negativity and Integrality:**  
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- **Index Set $I$:** All unique values in "Product Name" from table_id file_0_view_0 (PizzaSalesDataset.csv).
- **Parameter $A_i$:** "Revenue" column in table_id file_0_view_0.
- **Parameter $d_i$:** "Demand" column in table_id file_0_view_0.
- **Parameter $I_i$:** "Initial Inventory" column in table_id file_0_view_0.