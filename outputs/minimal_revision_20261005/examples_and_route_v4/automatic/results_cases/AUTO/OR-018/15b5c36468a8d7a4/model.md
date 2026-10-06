**Abstract Mathematical Model**

**Index Sets:**
- $I$: Set of ‘Baby’ products (from all records in file_0_view_0; indexed by Product Name).

**Parameters:**
- $r_i$: Revenue per unit of product $i$ (from Revenue column, file_0_view_0).
- $d_i$: Demand for product $i$ (from Demand column, file_0_view_0).
- $s_i$: Initial inventory of product $i$ (from Initial Inventory column, file_0_view_0).

**Decision Variables:**
- $x_i$: Number of units of product $i$ to fulfill, $x_i \in \mathbb{Z}_{\geq 0}$.

**Objective:**
\[
\max \sum_{i \in I} r_i x_i
\]

**Constraints:**
1. **Demand fulfillment:**  
   $\quad x_i \leq d_i \quad \forall i \in I$

2. **Inventory limit:**  
   $\quad x_i \leq s_i \quad \forall i \in I$

3. **Nonnegativity and integrality:**  
   $\quad x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I$

---

**Data Mapping**

- $I$: All records in file_0_view_0 (Product Name)
- $r_i$: file_0_view_0, column Revenue, key Product Name
- $d_i$: file_0_view_0, column Demand, key Product Name
- $s_i$: file_0_view_0, column Initial Inventory, key Product Name

**Variable mapping:**  
- $x_i$: Number of units of product $i$ to fulfill, for each $i \in I$ (Product Name from file_0_view_0)