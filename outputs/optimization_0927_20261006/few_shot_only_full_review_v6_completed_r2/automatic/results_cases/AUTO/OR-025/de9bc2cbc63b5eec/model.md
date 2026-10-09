---

### Abstract Mathematical Model

#### Index Sets
- $\mathcal{I}$: Set of all products in the dataset classified as 'TABLET' (i.e., all records where the value in column 'Product Name' begins with 'TABLET_').

#### Parameters
- $A_i$: Revenue per unit for product $i \in \mathcal{I}$, from column 'Revenue'.
- $d_i$: Total deterministic demand for product $i \in \mathcal{I}$, from column 'Demand'.
- $I_i$: Initial inventory for product $i \in \mathcal{I}$, from column 'Initial Inventory'.

#### Decision Variables
- $x_i \in \mathbb{Z}_+$: Number of units of product $i \in \mathcal{I}$ to fulfill (non-negative integer).

#### Objective
\[
\max \sum_{i \in \mathcal{I}} A_i x_i
\]

#### Constraints
1. **Inventory and Demand Fulfillment Bounds:**
   \[
   0 \leq x_i \leq \min\{I_i, d_i\} \qquad \forall i \in \mathcal{I}
   \]

#### Data Mapping

- **Source Table:** `file_0_view_0` (from `SmartphoneRetailOutletSalesData.csv`)
- **Index Set Selection:** $\mathcal{I}$ is the set of all records where `Product Name` (column) starts with the string `'TABLET_'`.
- **Parameters:**
  - $A_i$: `Revenue` column, for $i \in \mathcal{I}$
  - $d_i$: `Demand` column, for $i \in \mathcal{I}$
  - $I_i$: `Initial Inventory` column, for $i \in \mathcal{I}$
- **Decision Variables:** $x_i$ defined for all $i \in \mathcal{I}$

---

**End of Model.**