**Abstract Mathematical Model**

**Index Sets**
- $I$: Set of products, indexed by $i$ (from all "Product Name" in file_0_view_0).

**Parameters**
- $r_i$: Revenue per unit of product $i$ ("Revenue", file_0_view_0).
- $d_i$: Demand for product $i$ ("Demand", file_0_view_0).
- $s_i$: Initial inventory available for product $i$ ("Initial Inventory", file_0_view_0).

**Decision Variables**
- $x_i$: Number of units of product $i$ to fulfill (allocate to demand), $x_i \in \mathbb{Z}_{\geq 0}$.

**Objective**
\[
\max \sum_{i \in I} r_i x_i
\]

**Constraints**
1. **Demand fulfillment:**  
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
2. **Inventory limit:**  
   \[
   x_i \leq s_i \quad \forall i \in I
   \]
3. **Nonnegativity and integrality:**  
   \[
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
   \]

---

**Data Mapping**

- $I$: All records in table_id = file_0_view_0, column "Product Name"
- $r_i$: file_0_view_0, column "Revenue", keyed by "Product Name"
- $d_i$: file_0_view_0, column "Demand", keyed by "Product Name"
- $s_i$: file_0_view_0, column "Initial Inventory", keyed by "Product Name"