#### Index Sets

- Let $\mathcal{I}$ be the set of all products classified under ‘Aalop’ in the dataset (from column “Product Name” in table_id: file_0_view_0).

#### Parameters

- $A_i$: Revenue per unit of product $i$ (from column “Revenue” in table_id: file_0_view_0).
- $d_i$: Demand for product $i$ during the sales horizon (from column “Demand” in table_id: file_0_view_0).
- $I_i$: Initial inventory of product $i$ (from column “Initial Inventory” in table_id: file_0_view_0).

#### Decision Variables

- $x_i$: Number of units of product $i$ to fulfill, $\forall i \in \mathcal{I}$.

#### Objective

$$
\max \quad \sum_{i \in \mathcal{I}} A_i \cdot x_i
$$

#### Constraints

1. **Inventory Constraints:**
   $$
   x_i \leq I_i, \quad \forall i \in \mathcal{I}
   $$

2. **Demand Constraints:**
   $$
   x_i \leq d_i, \quad \forall i \in \mathcal{I}
   $$

3. **Variable Domain:**
   $$
   x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in \mathcal{I}
   $$

#### Data Mapping

- **table_id:** file_0_view_0
    - **Product Name**: Index set $\mathcal{I}$
    - **Revenue**: Parameter $A_i$
    - **Demand**: Parameter $d_i$
    - **Initial Inventory**: Parameter $I_i$