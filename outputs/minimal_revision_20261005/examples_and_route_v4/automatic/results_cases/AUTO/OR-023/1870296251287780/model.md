**Abstract Mathematical Model**

**Index Sets:**
- $I$: Set of products classified under ‘ELE-S’, indexed by $i$.  
  (Data: all Product_Reference in file_0_view_0)

**Parameters:**
- $r_i$: Revenue per unit of product $i$.  
  (Data: Revenue, file_0_view_0, column 'Revenue', key 'Product_Reference')
- $d_i$: Demand for product $i$.  
  (Data: Demand, file_0_view_0, column 'Demand', key 'Product_Reference')
- $s_i$: Initial inventory of product $i$.  
  (Data: Initial Inventory, file_0_view_0, column 'Initial Inventory', key 'Product_Reference')

**Decision Variables:**
- $x_i$: Number of units of product $i$ to fulfill.  
  Domain: $x_i \in \mathbb{Z}_{\geq 0}$, for all $i \in I$

---

**Objective:**
\[
\max \sum_{i \in I} r_i x_i
\]

**Constraints:**
1. **Inventory constraint:**  
  For all $i \in I$,
\[
x_i \leq s_i
\]
2. **Demand constraint:**  
  For all $i \in I$,
\[
x_i \leq d_i
\]
3. **Nonnegativity and integrality:**  
  For all $i \in I$,
\[
x_i \in \mathbb{Z}_{\geq 0}
\]

---

**Data Mapping**

| Symbol         | Source Table (table_id) | Column Name         | Key Column           |
|----------------|------------------------|---------------------|----------------------|
| $I$            | file_0_view_0          | Product_Reference   | —                    |
| $r_i$          | file_0_view_0          | Revenue             | Product_Reference    |
| $d_i$          | file_0_view_0          | Demand              | Product_Reference    |
| $s_i$          | file_0_view_0          | Initial Inventory   | Product_Reference    |

---

**Summary:**  
Choose $x_i$ for each product $i \in I$ to maximize total revenue, subject to not exceeding both initial inventory and demand, with $x_i$ nonnegative integers. All parameters and index sets are mapped directly from the supplied data.