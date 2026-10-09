**Sets:**

- $I$ : Index set of all products classified as "Baby" in table_id "file_0_view_0", column "Product Name".

**Parameters:**

- $A_i$ : Revenue per unit of product $i \in I$ (from "file_0_view_0", column "Revenue").
- $d_i$ : Demand for product $i \in I$ (from "file_0_view_0", column "Demand").
- $I_i$ : Initial inventory for product $i \in I$ (from "file_0_view_0", column "Initial Inventory").

**Decision Variables:**

- $x_i$ : Number of units of product $i \in I$ to fulfill; $x_i \in \mathbb{Z}_+$.

**Objective:**

$$
\max \sum_{i \in I} A_i \cdot x_i
$$

**Constraints:**

1. **Inventory Constraint:**
   $$
   x_i \leq I_i, \quad \forall i \in I
   $$
2. **Demand Constraint:**
   $$
   x_i \leq d_i, \quad \forall i \in I
   $$
3. **Non-negativity and Integrality:**
   $$
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   $$

**Data Mapping:**

- Table: "file_0_view_0"
    - Product Name $\rightarrow$ index set $I$ (filtered for "Baby" products)
    - Revenue $\rightarrow$ parameter $A_i$
    - Demand $\rightarrow$ parameter $d_i$
    - Initial Inventory $\rightarrow$ parameter $I_i$