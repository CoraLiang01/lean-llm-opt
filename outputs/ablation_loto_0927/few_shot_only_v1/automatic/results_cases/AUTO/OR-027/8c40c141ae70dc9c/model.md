#### Index Sets

- Let $\mathcal{I}$ be the set of all products in the dataset with "Organ" in their 'Sub Category' name, i.e.,
  $$
  \mathcal{I} = \{ i : \text{‘Sub Category’ of product } i \text{ contains the substring "Organ"} \}
  $$

#### Parameters

- $A_i$: Revenue per unit of product $i$ (from column ‘Revenue’ in table_id = file_0_view_0)
- $d_i$: Demand for product $i$ (from column ‘Demand’ in table_id = file_0_view_0)
- $I_i$: Initial inventory of product $i$ (from column ‘Initial Inventory’ in table_id = file_0_view_0)

#### Decision Variables

- $x_i$: Number of units of product $i$ to fulfill, $\forall i \in \mathcal{I}$

#### Objective

$$
\max \sum_{i \in \mathcal{I}} A_i \cdot x_i
$$

#### Constraints

1. **Inventory Constraint** (cannot fulfill more than available inventory):
   $$
   x_i \leq I_i, \quad \forall i \in \mathcal{I}
   $$

2. **Demand Constraint** (cannot fulfill more than demand):
   $$
   x_i \leq d_i, \quad \forall i \in \mathcal{I}
   $$

3. **Non-negativity and Integrality**:
   $$
   x_i \in \mathbb{Z}_+, \quad \forall i \in \mathcal{I}
   $$

#### Data Mapping

- **table_id:** file_0_view_0
- **Columns used:**
  - ‘Sub Category’ (for identifying $\mathcal{I}$: all rows where value contains "Organ")
  - ‘Revenue’ (parameter $A_i$)
  - ‘Demand’ (parameter $d_i$)
  - ‘Initial Inventory’ (parameter $I_i$)