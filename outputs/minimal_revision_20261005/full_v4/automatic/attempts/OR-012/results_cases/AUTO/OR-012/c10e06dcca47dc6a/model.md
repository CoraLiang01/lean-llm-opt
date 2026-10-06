#### Sets

- $I$: Set of all products, indexed by $i$.

#### Parameters

- $A_i$: Revenue per unit of product $i$.  
  (from column "Revenue")
- $d_i$: Total deterministic demand for product $i$ over the sales horizon.  
  (from column "Demand")
- $I_i$: Initial inventory available for product $i$.  
  (from column "Initial Inventory")

#### Decision Variables

- $x_i$: Number of units of product $i$ to fulfill for customer purchases.  
  ($x_i \in \mathbb{Z}_+, \; 0 \leq x_i \leq \min\{d_i, I_i\}$)

#### Objective

$$
\max \sum_{i \in I} A_i \cdot x_i
$$

#### Constraints

1. **Demand fulfillment cannot exceed demand:**
   $$
   x_i \leq d_i \quad \forall i \in I
   $$
2. **Cannot sell more than available inventory:**
   $$
   x_i \leq I_i \quad \forall i \in I
   $$
3. **Non-negativity and integrality:**
   $$
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   $$

---

#### Data Mapping

- **Table:** `file_0_view_0` (from `OnlineSalesDataset.csv`)
- **Columns:**
  - Product Name: index set $I$
  - Revenue: parameter $A_i$
  - Demand: parameter $d_i$
  - Initial Inventory: parameter $I_i$