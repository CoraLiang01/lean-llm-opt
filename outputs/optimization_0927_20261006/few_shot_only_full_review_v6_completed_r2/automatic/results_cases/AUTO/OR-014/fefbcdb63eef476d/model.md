---

### Abstract Mathematical Model

#### Index Sets
- Let $\mathcal{I}$ be the set of all pizza types, indexed by $i$.

#### Parameters
- $A_i$: Revenue per unit for pizza type $i$.
- $d_i$: Demand for pizza type $i$.
- $I_i$: Initial Inventory for pizza type $i$.

#### Decision Variables
- $x_i \in \mathbb{Z}_+, \quad \forall i \in \mathcal{I}$  
  (Number of units of pizza type $i$ to fulfill)

#### Objective
\[
\max \sum_{i \in \mathcal{I}} A_i \cdot x_i
\]

#### Constraints
1. **Inventory and Demand Constraints**  
   For all $i \in \mathcal{I}$:
   \[
   x_i \leq I_i
   \]
   \[
   x_i \leq d_i
   \]
2. **Non-negativity and Integrality**
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in \mathcal{I}
   \]

---

### Data Mapping

- **Source Table:**  
  - `file_0_view_0` (from `PizzaSalesDataset.csv`)
- **Index Set:**  
  - $\mathcal{I}$: All records in column `Product Name`
- **Parameters:**  
  - $A_i$: Value from column `Revenue` for pizza $i$
  - $d_i$: Value from column `Demand` for pizza $i$
  - $I_i$: Value from column `Initial Inventory` for pizza $i$
- **Selection:**  
  - All records in the table are included; no filtering is applied.

---