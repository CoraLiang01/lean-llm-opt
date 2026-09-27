#### Abstract Mathematical Optimization Model

**Index Sets:**

- $I$ : Set of all products.

**Parameters:**

- $A_i$ : Revenue per unit of product $i \in I$ (from column ‘Revenue’).
- $d_i$ : Total demand for product $i \in I$ (from column ‘Demand’).
- $I_i$ : Initial inventory for product $i \in I$ (from column ‘Initial Inventory’).

**Decision Variables:**

- $x_i$ : Number of units of product $i \in I$ to fulfill for customer purchases.

**Objective:**

$$
\max \sum_{i \in I} A_i \cdot x_i
$$

**Constraints:**

1. **Demand fulfillment constraint:**
   $$
   x_i \leq d_i, \quad \forall i \in I
   $$

2. **Inventory constraint:**
   $$
   x_i \leq I_i, \quad \forall i \in I
   $$

3. **Non-negativity and integrality:**
   $$
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   $$

---

#### Data Mapping

- **Table:** `file_0_view_0` (from `OnlineSalesDataset.csv`)
- **Columns:**
    - Product Name $\rightarrow$ index set $I$
    - Revenue $\rightarrow$ parameter $A_i$
    - Demand $\rightarrow$ parameter $d_i$
    - Initial Inventory $\rightarrow$ parameter $I_i$