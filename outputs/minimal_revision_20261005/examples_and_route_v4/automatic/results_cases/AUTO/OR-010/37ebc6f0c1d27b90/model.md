**Abstract Mathematical Model**

**Index Sets**
- $I$: Set of products (from all "Product Name" in table_id: file_0_view_0)

**Parameters**
- $r_i$: Revenue per unit of product $i$ (from "Revenue", table_id: file_0_view_0, key: "Product Name")
- $d_i$: Demand for product $i$ (from "Demand", table_id: file_0_view_0, key: "Product Name")
- $s_i$: Initial inventory for product $i$ (from "Initial Inventory", table_id: file_0_view_0, key: "Product Name")

**Decision Variables**
- $x_i$: Number of units of product $i$ fulfilled (orders fulfilled), $x_i \in \mathbb{Z}_{\geq 0}$

**Objective**
\[
\max \sum_{i \in I} r_i x_i
\]

**Constraints**
1. **Inventory constraint:** 
   \[
   x_i \leq s_i \qquad \forall i \in I
   \]
2. **Demand constraint:** 
   \[
   x_i \leq d_i \qquad \forall i \in I
   \]
3. **Non-negativity and integrality:** 
   \[
   x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I
   \]

---

**Data Mapping**

- $I$: All records in table_id: file_0_view_0, column "Product Name"
- $r_i$: table_id: file_0_view_0, columns "Product Name", "Revenue"
- $d_i$: table_id: file_0_view_0, columns "Product Name", "Demand"
- $s_i$: table_id: file_0_view_0, columns "Product Name", "Initial Inventory"