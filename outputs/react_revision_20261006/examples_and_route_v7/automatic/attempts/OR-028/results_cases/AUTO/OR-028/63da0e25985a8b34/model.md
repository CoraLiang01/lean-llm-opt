#### Mathematical Model

Let $I$ be the set of products, indexed by $i$ (with business identifier Product Name from file_0_view_0).

**Parameters:**
- $r_i$: Revenue per unit of product $i$ (Revenue, file_0_view_0)
- $d_i$: Demand for product $i$ (Demand, file_0_view_0)
- $s_i$: Initial inventory of product $i$ (Initial Inventory, file_0_view_0)

**Decision Variables:**
- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_{\geq 0}$

**Objective:**
$$
\max \sum_{i \in I} r_i x_i
$$

**Constraints:**
1. **Demand fulfillment cannot exceed demand:**
   $$
   x_i \leq d_i \quad \forall i \in I
   $$
2. **Cannot fulfill more than available inventory:**
   $$
   x_i \leq s_i \quad \forall i \in I
   $$
3. **Nonnegativity and integrality:**
   $$
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
   $$

---

#### Data Mapping

- $I$: All records in file_0_view_0, column Product Name
- $r_i$: file_0_view_0, column Revenue, keyed by Product Name
- $d_i$: file_0_view_0, column Demand, keyed by Product Name
- $s_i$: file_0_view_0, column Initial Inventory, keyed by Product Name

- Decision variable $x_i$ is defined for each $i \in I$ (Product Name, file_0_view_0) and represents the number of units of product $i$ to fulfill.