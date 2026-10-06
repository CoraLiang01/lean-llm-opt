#### Abstract Mathematical Model

**Index Sets:**
- $I$: Set of “4U” products (indexed by $i$; see Data Mapping for identifiers).

**Parameters:**
- $r_i$: Revenue per unit of product $i$.
- $d_i$: Demand for product $i$ during the sales horizon.
- $s_i$: Initial inventory of product $i$.

**Decision Variables:**
- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_{\geq 0}$.

**Objective:**
\[
\max \sum_{i \in I} r_i x_i
\]

**Constraints:**
1. **Inventory and Demand Fulfillment:**
   \[
   0 \leq x_i \leq \min\{d_i, s_i\} \quad \forall i \in I
   \]
   (Equivalently, two constraints per product:)
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
   \[
   x_i \leq s_i \quad \forall i \in I
   \]
2. **Integrality:**
   \[
   x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
   \]

---

#### Data Mapping

| Index $i$ | Product Name         | $r_i$ (Revenue) | $d_i$ (Demand) | $s_i$ (Initial Inventory) | table_id:column_name (source_row) |
|-----------|---------------------|-----------------|---------------|--------------------------|-----------------------------------|
| 1         | 4U_Service_22       | 56.0            | 5             | 30                       | file_0_view_0:Product Name, Revenue, Demand, Initial Inventory (1) |
| 2         | 4U_Service_36       | 21.6            | 3             | 20                       | file_0_view_0:Product Name, Revenue, Demand, Initial Inventory (2) |
| 3         | 4U_Service_7        | 62.5            | 3             | 20                       | file_0_view_0:Product Name, Revenue, Demand, Initial Inventory (3) |

- All parameters ($r_i$, $d_i$, $s_i$) are mapped from columns “Revenue”, “Demand”, and “Initial Inventory” in table_id file_0_view_0, preserving source_row order and identifiers.

---

**Summary:**  
Maximize total revenue from “4U” products by choosing integer fulfillment quantities $x_i$ for each product $i$, subject to both inventory and demand limits for each product. All data and identifiers are mapped directly from the supplied source.