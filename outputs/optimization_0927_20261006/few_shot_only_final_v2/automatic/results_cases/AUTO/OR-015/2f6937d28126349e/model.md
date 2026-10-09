**Sets:**  
- $I$: Index set of all products classified under ‘Aalop’.

**Parameters:**  
- $A_i$: Revenue per unit of product $i \in I$.  
- $d_i$: Demand for product $i \in I$.  
- $I_i$: Initial inventory for product $i \in I$.

**Decision Variables:**  
- $x_i$: Number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+, \forall i \in I$.

**Objective:**  
$$
\max \sum_{i \in I} A_i x_i
$$

**Constraints:**  
1. **Inventory constraint:**  
   $$
   x_i \leq I_i, \quad \forall i \in I
   $$
2. **Demand constraint:**  
   $$
   x_i \leq d_i, \quad \forall i \in I
   $$
3. **Non-negativity and integrality:**  
   $$
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   $$

---

**Data Mapping:**  
- Source table: `file_0_view_0`  
- Product index set $I$: All records in `file_0_view_0` where "Product Name" is classified under ‘Aalop’.  
- $A_i$: `file_0_view_0`, column "Revenue"  
- $d_i$: `file_0_view_0`, column "Demand"  
- $I_i$: `file_0_view_0`, column "Initial Inventory"