**Abstract Mathematical Model**

**Index Set:**
- $I$: Set of dairy products, indexed by $i$ (from all Full_Product_Name in file_0_view_0).

**Parameters:**
- $r_i$: Revenue per unit of product $i$ (Revenue, file_0_view_0, column "Revenue").
- $d_i$: Demand for product $i$ (Demand, file_0_view_0, column "Demand").
- $s_i$: Initial inventory for product $i$ (Initial Inventory, file_0_view_0, column "Initial Inventory").

**Decision Variables:**
- $x_i$: Number of units of product $i$ to fulfill. ($x_i \in \mathbb{Z}_{\geq 0}$)

**Objective:**
\[
\max \sum_{i \in I} r_i x_i
\]

**Constraints:**
1. **Demand fulfillment:**  
   \[
   x_i \leq d_i \qquad \forall i \in I
   \]
2. **Inventory limit:**  
   \[
   x_i \leq s_i \qquad \forall i \in I
   \]
3. **Nonnegativity and integrality:**  
   \[
   x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I
   \]

---

**Data Mapping**

- $I$: All records in file_0_view_0, column "Full_Product_Name"
- $r_i$: file_0_view_0, column "Revenue", keyed by "Full_Product_Name"
- $d_i$: file_0_view_0, column "Demand", keyed by "Full_Product_Name"
- $s_i$: file_0_view_0, column "Initial Inventory", keyed by "Full_Product_Name"