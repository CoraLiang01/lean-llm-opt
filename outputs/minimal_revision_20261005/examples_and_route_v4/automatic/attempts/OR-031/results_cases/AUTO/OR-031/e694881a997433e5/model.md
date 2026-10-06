**Mathematical Model**

**Index Sets:**
- $I$: Set of dairy products, indexed by $i$.  
 (Data: all Full_Product_Name in file_0_view_0)

**Parameters:**
- $r_i$: Revenue per unit of product $i$.  
 (Data: Revenue, file_0_view_0, column 'Revenue', key 'Full_Product_Name')
- $d_i$: Demand for product $i$.  
 (Data: Demand, file_0_view_0, column 'Demand', key 'Full_Product_Name')
- $s_i$: Initial inventory for product $i$.  
 (Data: Initial Inventory, file_0_view_0, column 'Initial Inventory', key 'Full_Product_Name')

**Decision Variables:**
- $x_i$: Number of units of product $i$ to fulfill.  
 Domain: $x_i \in \mathbb{Z}_{\geq 0}$

**Objective:**
\[
\max \sum_{i \in I} r_i x_i
\]

**Constraints:**
1. **Demand fulfillment:**  
 $\forall i \in I: \quad x_i \leq d_i$

2. **Inventory limit:**  
 $\forall i \in I: \quad x_i \leq s_i$

3. **Nonnegativity and integrality:**  
 $\forall i \in I: \quad x_i \in \mathbb{Z}_{\geq 0}$

---

**Data Mapping**

- $I$: All records in file_0_view_0, column 'Full_Product_Name'
- $r_i$: file_0_view_0, columns 'Full_Product_Name', 'Revenue'
- $d_i$: file_0_view_0, columns 'Full_Product_Name', 'Demand'
- $s_i$: file_0_view_0, columns 'Full_Product_Name', 'Initial Inventory'

**Variable:**  
- $x_i$: Number of units of product $i$ to fulfill (nonnegative integer), indexed by 'Full_Product_Name' in file_0_view_0