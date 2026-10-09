**Sets:**  
Let $\mathcal{I}$ be the set of all products in table_id `file_0_view_0` where `Product Name` contains the substring "FAUX".

**Parameters:**  
For each $i \in \mathcal{I}$:
- $A_i$: Revenue per unit of product $i$ (`Revenue` from `file_0_view_0`)
- $d_i$: Deterministic demand for product $i$ (`Demand` from `file_0_view_0`)
- $I_i$: Initial inventory for product $i$ (`Initial Inventory` from `file_0_view_0`)

**Decision Variables:**  
For each $i \in \mathcal{I}$:
- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_+$

**Objective:**  
Maximize total revenue:
$$
\max \sum_{i \in \mathcal{I}} A_i \cdot x_i
$$

**Constraints:**  
For all $i \in \mathcal{I}$:
1. Inventory constraint:  
   $$
   x_i \leq I_i
   $$
2. Demand constraint:  
   $$
   x_i \leq d_i
   $$
3. Non-negativity and integrality:  
   $$
   x_i \in \mathbb{Z}_+, \quad \forall i \in \mathcal{I}
   $$

---

**Data Mapping:**  
- Source Table: `file_0_view_0` (from `ZARASales.csv`)
- Product Name: `Product Name`
- Revenue per unit: `Revenue`
- Demand: `Demand`
- Initial Inventory: `Initial Inventory`