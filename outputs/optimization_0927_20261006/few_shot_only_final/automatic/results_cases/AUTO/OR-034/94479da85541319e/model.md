---

### Abstract Mathematical Model

#### Index Sets
- $I$: Set of all products, indexed by $i$.

#### Parameters
- $A_i$: Revenue per unit of product $i$.  
- $d_i$: Demand for product $i$.  
- $I_i$: Initial inventory for product $i$.

#### Decision Variables
- $x_i$: Quantity of product $i$ to fulfill (sell), $\forall i \in I$.

#### Objective
$$
\max \sum_{i \in I} A_i \cdot x_i
$$

#### Constraints
1. **Demand and Inventory Fulfillment Bounds:**  
   $$
   0 \leq x_i \leq \min\{d_i, I_i\}, \quad \forall i \in I
   $$

2. **Variable Domain:**  
   $$
   x_i \in \mathbb{R}_+, \quad \forall i \in I
   $$

---

### Data Mapping

- **Table:** `file_0_view_0` (from `Frenchbakerydailysales.csv`)
- **Columns:**
    - Product Name: Index set $I$
    - Revenue: Parameter $A_i$
    - Demand: Parameter $d_i$
    - Initial Inventory: Parameter $I_i$

---