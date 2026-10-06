**Abstract Mathematical Model**

**Index Sets**
- $P$: Set of products, indexed by $i$. (All "Product Name" in file_0_view_0.)

**Parameters**
- $r_i$: Revenue per unit of product $i$. (file_0_view_0, column "Revenue", key "Product Name")
- $d_i$: Demand for product $i$. (file_0_view_0, column "Demand", key "Product Name")
- $s_i$: Initial inventory for product $i$. (file_0_view_0, column "Initial Inventory", key "Product Name")

**Decision Variables**
- $x_i$: Number of units of product $i$ to fulfill (allocate to demand), $x_i \in \mathbb{Z}_{\geq 0}$

**Objective**
\[
\max \sum_{i \in P} r_i x_i
\]

**Constraints**
1. **Demand fulfillment cannot exceed demand:**
   \[
   x_i \leq d_i \quad \forall i \in P
   \]
2. **Cannot allocate more than available inventory:**
   \[
   x_i \leq s_i \quad \forall i \in P
   \]
3. **Nonnegativity and integrality:**
   \[
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in P
   \]

---

**Data Mapping**

- $P$: All records in file_0_view_0, column "Product Name"
- $r_i$: file_0_view_0, columns "Product Name", "Revenue"
- $d_i$: file_0_view_0, columns "Product Name", "Demand"
- $s_i$: file_0_view_0, columns "Product Name", "Initial Inventory"