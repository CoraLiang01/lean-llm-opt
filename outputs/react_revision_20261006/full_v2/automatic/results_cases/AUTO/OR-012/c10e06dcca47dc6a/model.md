##### Symbolic Mathematical Model

**Index Set:**
- $i \in \mathcal{P}$: Set of all products in the dataset.

**Parameters:**
- $A_i$: Revenue per unit of product $i$ (from column "Revenue").
- $d_i$: Total demand for product $i$ over the sales horizon (from column "Demand").
- $I_i$: Initial inventory of product $i$ (from column "Initial Inventory").

**Decision Variables:**
- $x_i$: Number of units of product $i$ to fulfill for customer purchases.
- Variable domain: $x_i \in \mathbb{Z}_+, \quad \forall i \in \mathcal{P}$

**Objective:**
\[
\max \sum_{i \in \mathcal{P}} A_i \cdot x_i
\]

**Constraints:**
1. **Inventory Constraint:**
   \[
   x_i \leq I_i, \quad \forall i \in \mathcal{P}
   \]
2. **Demand Constraint:**
   \[
   x_i \leq d_i, \quad \forall i \in \mathcal{P}
   \]
3. **Non-negativity and Integrality:**
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in \mathcal{P}
   \]

##### Data Mapping

- Table: `file_0_view_0` (from `OnlineSalesDataset.csv`)
    - Product set $\mathcal{P}$: All unique values in column "Product Name"
    - Revenue parameter $A_i$: Column "Revenue"
    - Demand parameter $d_i$: Column "Demand"
    - Initial inventory parameter $I_i$: Column "Initial Inventory"