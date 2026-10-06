#### Index Sets

- $I$: set of all products, indexed by $i$ (from column "Product Name" in table_id: file_0_view_0)

#### Parameters

- $A_i$: revenue per unit of product $i$ (from column "Revenue" in table_id: file_0_view_0)
- $d_i$: total demand for product $i$ (from column "Demand" in table_id: file_0_view_0)
- $I_i$: initial inventory of product $i$ (from column "Initial Inventory" in table_id: file_0_view_0)

#### Decision Variables

- $x_i$: number of units of product $i$ to fulfill, $\forall i \in I$  
  (domain: $x_i \in \mathbb{Z}_+, \ x_i \geq 0$)

#### Objective

$$
\max \sum_{i \in I} A_i \cdot x_i
$$

#### Constraints

1. **Demand fulfillment cannot exceed demand:**
   $$
   x_i \leq d_i, \quad \forall i \in I
   $$

2. **Fulfillment cannot exceed initial inventory:**
   $$
   x_i \leq I_i, \quad \forall i \in I
   $$

3. **Non-negativity and integrality:**
   $$
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   $$

---

#### Data Mapping

- **table_id:** file_0_view_0
    - **Product Name** $\rightarrow$ index set $I$
    - **Revenue** $\rightarrow$ parameter $A_i$
    - **Demand** $\rightarrow$ parameter $d_i$
    - **Initial Inventory** $\rightarrow$ parameter $I_i$