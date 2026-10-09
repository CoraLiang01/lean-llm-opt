---

#### Index Sets

- Let $\mathcal{I}$ be the set of all pizza types, indexed by $i$.

#### Parameters

- $A_i$: Revenue per unit of pizza type $i$ (from column "Revenue", table_id: file_0_view_0)
- $d_i$: Demand for pizza type $i$ (from column "Demand", table_id: file_0_view_0)
- $I_i$: Initial inventory for pizza type $i$ (from column "Initial Inventory", table_id: file_0_view_0)

#### Decision Variables

- $x_i$: Number of units of pizza type $i$ to fulfill, $\forall i \in \mathcal{I}$

#### Objective

$$
\max \sum_{i \in \mathcal{I}} A_i \cdot x_i
$$

#### Constraints

1. **Inventory and Demand Fulfillment:**
   $$
   0 \leq x_i \leq \min\{I_i,\, d_i\}, \quad \forall i \in \mathcal{I}
   $$

2. **Integrality:**
   $$
   x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in \mathcal{I}
   $$

---

#### Data Mapping

- All parameters $A_i$, $d_i$, $I_i$ are mapped from table_id: file_0_view_0, columns "Revenue", "Demand", and "Initial Inventory", respectively. The index set $\mathcal{I}$ corresponds to all unique values in "Product Name" from the same table.