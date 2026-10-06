**Sets**  
Let $\mathcal{I}$ be the set of all products classified under ‘Books’ (from column Product_Name in table_id file_0_view_0).

**Parameters**  
For each $i \in \mathcal{I}$:
- $A_i$: Revenue per unit of product $i$ (from Revenue, file_0_view_0)
- $d_i$: Demand for product $i$ (from Demand, file_0_view_0)
- $I_i$: Initial inventory of product $i$ (from Initial Inventory, file_0_view_0)

**Decision Variables**  
For each $i \in \mathcal{I}$:
- $x_i$: Number of units of product $i$ to fulfill  
  Domain: $x_i \in \mathbb{Z}_+, \; 0 \leq x_i \leq \min\{d_i, I_i\}$

**Objective**  
Maximize total revenue:
$$
\max \sum_{i \in \mathcal{I}} A_i x_i
$$

**Constraints**
1. **Demand fulfillment:**  
  $x_i \leq d_i \quad \forall i \in \mathcal{I}$

2. **Inventory limit:**  
  $x_i \leq I_i \quad \forall i \in \mathcal{I}$

3. **Non-negativity and integrality:**  
  $x_i \in \mathbb{Z}_+, \quad \forall i \in \mathcal{I}$

---

**Data Mapping**

- **table_id:** file_0_view_0  
- **Product_Name:** Index set $\mathcal{I}$ (all products classified under ‘Books’)  
- **Revenue:** Parameter $A_i$  
- **Demand:** Parameter $d_i$  
- **Initial Inventory:** Parameter $I_i$