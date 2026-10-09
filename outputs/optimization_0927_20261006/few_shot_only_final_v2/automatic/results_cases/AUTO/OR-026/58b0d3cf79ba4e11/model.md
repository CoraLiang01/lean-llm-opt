---

**Sets:**  
- $\mathcal{F}$: Set of all products classified under ‘Fashion’ in the dataset.

**Parameters:**  
- $A_i$: Revenue per unit for product $i \in \mathcal{F}$ (from column ‘Revenue’).
- $d_i$: Total deterministic demand for product $i \in \mathcal{F}$ (from column ‘Demand’).
- $I_i$: Initial inventory for product $i \in \mathcal{F}$ (from column ‘Initial Inventory’).

**Decision Variables:**  
- $x_i$: Number of units of product $i \in \mathcal{F}$ to fulfill,  
  $x_i \in \mathbb{Z}_+$ (non-negative integers).

**Objective:**  
\[
\max \sum_{i \in \mathcal{F}} A_i \cdot x_i
\]

**Constraints:**  
1. **Inventory constraint:**  
   \[
   x_i \leq I_i \qquad \forall i \in \mathcal{F}
   \]
2. **Demand constraint:**  
   \[
   x_i \leq d_i \qquad \forall i \in \mathcal{F}
   \]
3. **Non-negativity and integrality:**  
   \[
   x_i \in \mathbb{Z}_+, \qquad \forall i \in \mathcal{F}
   \]

---

**Data Mapping:**  
- Source Table: `file_0_view_0`  
- Columns:  
  - Product Name: `Product Name`  
  - Revenue: `Revenue`  
  - Demand: `Demand`  
  - Initial Inventory: `Initial Inventory`  

---