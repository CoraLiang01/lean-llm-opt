**Abstract Mathematical Model**

**Index Sets:**
- $P$: Set of products (from `file_0_view_0`, column `Product Name`)

**Parameters:**
- $r_p$: Revenue per unit of product $p$ (from `file_0_view_0`, column `Revenue`)
- $d_p$: Demand for product $p$ (from `file_0_view_0`, column `Demand`)
- $s_p$: Initial inventory for product $p$ (from `file_0_view_0`, column `Initial Inventory`)

**Decision Variables:**
- $x_p$: Number of units of product $p$ to fulfill for customer purchases  
  Domain: $x_p \in \mathbb{Z}_{\geq 0}$, for all $p \in P$

**Objective:**
\[
\max \sum_{p \in P} r_p \, x_p
\]

**Constraints:**
1. **Demand fulfillment:**  
  $x_p \leq d_p \quad \forall p \in P$

2. **Inventory availability:**  
  $x_p \leq s_p \quad \forall p \in P$

3. **Nonnegativity and integrality:**  
  $x_p \in \mathbb{Z}_{\geq 0} \quad \forall p \in P$

---

**Data Mapping**

- $P$: All records in `file_0_view_0`, column `Product Name`
- $r_p$: `file_0_view_0`, column `Revenue`, keyed by `Product Name`
- $d_p$: `file_0_view_0`, column `Demand`, keyed by `Product Name`
- $s_p$: `file_0_view_0`, column `Initial Inventory`, keyed by `Product Name`