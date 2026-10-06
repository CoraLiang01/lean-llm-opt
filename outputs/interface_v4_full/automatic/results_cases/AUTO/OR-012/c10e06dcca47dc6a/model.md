#### Abstract Mathematical Optimization Model

**Index Sets:**
- $I$: Set of all products (indexed by $i$).

**Parameters:**
- $A_i$: Revenue per unit of product $i$.  
- $d_i$: Deterministic total demand for product $i$ over the sales horizon.
- $I_i$: Initial inventory available for product $i$.

**Decision Variables:**
- $x_i$: Number of units of product $i$ to fulfill for customer purchases.  
  Domain: $x_i \in \mathbb{Z}_+, \quad \forall i \in I$

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. **Inventory Constraints:**  
   \[
   x_i \leq I_i, \quad \forall i \in I
   \]
2. **Demand Constraints:**  
   \[
   x_i \leq d_i, \quad \forall i \in I
   \]
3. **Non-negativity and Integrality:**  
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

**Data Mapping:**  
- Table: `file_0_view_0` (from `OnlineSalesDataset.csv`)
    - Product identifier: `Product Name`
    - Revenue per unit: `Revenue`
    - Demand: `Demand`
    - Initial inventory: `Initial Inventory`