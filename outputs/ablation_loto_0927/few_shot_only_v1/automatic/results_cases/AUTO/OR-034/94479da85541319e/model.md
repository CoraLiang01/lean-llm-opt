#### Index Sets

- Let $\mathcal{I}$ be the set of all baked goods, indexed by $i$.

#### Parameters

- $A_i$: Revenue per unit of baked good $i$ (from column ‘Revenue’ in table_id: file_0_view_0).
- $d_i$: Total demand for baked good $i$ (from column ‘Demand’ in table_id: file_0_view_0).
- $I_i$: Initial inventory for baked good $i$ (from column ‘Initial Inventory’ in table_id: file_0_view_0).

#### Decision Variables

- $x_i$: Number of units of baked good $i$ to fulfill, $\forall i \in \mathcal{I}$.

#### Objective

$$
\max \sum_{i \in \mathcal{I}} A_i \cdot x_i
$$

#### Constraints

1. **Demand fulfillment constraint:**
   $$
   x_i \leq d_i, \quad \forall i \in \mathcal{I}
   $$

2. **Inventory constraint:**
   $$
   x_i \leq I_i, \quad \forall i \in \mathcal{I}
   $$

3. **Non-negativity and integrality:**
   $$
   x_i \in \mathbb{Z}_+, \quad \forall i \in \mathcal{I}
   $$

---

#### Data Mapping

- All parameters ($A_i$, $d_i$, $I_i$) are mapped from table_id: file_0_view_0, columns: ‘Revenue’, ‘Demand’, ‘Initial Inventory’ in the source Frenchbakerydailysales.csv. The index set $\mathcal{I}$ corresponds to all unique values in column ‘Product Name’ of the same table.