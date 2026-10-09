**Abstract Mathematical Model**

**Index Sets:**  
- $I$: Set of all pizza types (indexed by $i$).

**Parameters:**  
- $A_i$: Revenue per unit of pizza type $i$.  
- $d_i$: Demand for pizza type $i$.  
- $I_i$: Initial inventory available for pizza type $i$.

**Decision Variables:**  
- $x_i$: Number of units of pizza type $i$ to fulfill, $x_i \in \mathbb{Z}_+, \forall i \in I$.

**Objective:**  
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**  
1. **Inventory and Demand Fulfillment Bounds:**  
   \[
   0 \leq x_i \leq \min\{d_i, I_i\}, \quad \forall i \in I
   \]
   (Equivalently, two separate constraints:)
   \[
   x_i \leq d_i, \quad \forall i \in I
   \]
   \[
   x_i \leq I_i, \quad \forall i \in I
   \]
2. **Integrality:**  
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

**Data Mapping:**  
- Table: `file_0_view_0`  
  - Pizza types: `Product Name`  
  - Revenue per unit: `Revenue`  
  - Demand: `Demand`  
  - Initial inventory: `Initial Inventory`