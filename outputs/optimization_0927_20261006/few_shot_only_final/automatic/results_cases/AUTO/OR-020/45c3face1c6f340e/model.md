#### Index Sets
- Let $\mathcal{I}$ be the set of all products, indexed by $i$.

#### Parameters
- $A_i$: Revenue per unit of product $i$ (from column ‘Revenue’ in table_id: file_0_view_0)
- $d_i$: Demand for product $i$ (from column ‘Demand’ in table_id: file_0_view_0)
- $I_i$: Initial inventory of product $i$ (from column ‘Initial Inventory’ in table_id: file_0_view_0)

#### Decision Variables
- $x_i$: Number of units of product $i$ to fulfill, $\forall i \in \mathcal{I}$

#### Objective
$$
\max \sum_{i \in \mathcal{I}} A_i \cdot x_i
$$

#### Constraints
1. **Demand and Inventory Fulfillment Bounds:**
   $$
   0 \leq x_i \leq \min\{d_i, I_i\}, \quad \forall i \in \mathcal{I}
   $$
   (Or, equivalently, two constraints per product:)
   $$
   x_i \leq d_i, \quad \forall i \in \mathcal{I}
   $$
   $$
   x_i \leq I_i, \quad \forall i \in \mathcal{I}
   $$
   $$
   x_i \geq 0, \quad \forall i \in \mathcal{I}
   $$

2. **Variable Domain:**
   $$
   x_i \in \mathbb{Z}_+, \quad \forall i \in \mathcal{I}
   $$

---

**Data Mapping:**  
- Table: file_0_view_0 (from SalesDatainBusinesses.csv)  
- Columns used:  
  - ‘Product Name’ (for index set $\mathcal{I}$)  
  - ‘Revenue’ (for $A_i$)  
  - ‘Demand’ (for $d_i$)  
  - ‘Initial Inventory’ (for $I_i$)