#### Index Sets
- Let $\mathcal{I}$ be the set of all products, indexed by $i$.

#### Parameters
- $A_i$: Revenue per unit of product $i$ (from column ‘Revenue’).
- $d_i$: Total demand for product $i$ over the sales horizon (from column ‘Demand’).
- $I_i$: Initial inventory available for product $i$ (from column ‘Initial Inventory’).

#### Decision Variables
- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_{\geq 0}$, for all $i \in \mathcal{I}$.

#### Objective
\[
\max \sum_{i \in \mathcal{I}} A_i \cdot x_i
\]

#### Constraints
1. **Demand fulfillment cannot exceed demand:**
   \[
   x_i \leq d_i, \quad \forall i \in \mathcal{I}
   \]
2. **Cannot fulfill more than available inventory:**
   \[
   x_i \leq I_i, \quad \forall i \in \mathcal{I}
   \]
3. **Non-negativity and integrality:**
   \[
   x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in \mathcal{I}
   \]

---

#### Data Mapping

- **Source Table:** file_0_view_0 (WomenClothingEcommerceSalesData.csv)
- **Index Set:** $\mathcal{I}$ corresponds to all rows in file_0_view_0.
- **Parameters:**
  - $A_i$ from column ‘Revenue’
  - $d_i$ from column ‘Demand’
  - $I_i$ from column ‘Initial Inventory’
- **No filters**: All records in file_0_view_0 are included.