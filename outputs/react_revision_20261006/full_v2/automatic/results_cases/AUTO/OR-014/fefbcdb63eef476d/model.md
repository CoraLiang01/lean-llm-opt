#### Mathematical Optimization Model

**Index Set:**  
Let $\mathcal{I}$ be the set of all pizza types in the dataset.

**Parameters:**  
For each $i \in \mathcal{I}$:
- $A_i$: Revenue per unit of pizza type $i$ (from column "Revenue")
- $d_i$: Demand for pizza type $i$ (from column "Demand")
- $I_i$: Initial inventory for pizza type $i$ (from column "Initial Inventory")

**Decision Variables:**  
For each $i \in \mathcal{I}$:
- $x_i$: Number of units of pizza type $i$ to fulfill, $x_i \in \mathbb{Z}_+, \forall i \in \mathcal{I}$

**Objective:**  
$$
\max \sum_{i \in \mathcal{I}} A_i x_i
$$

**Constraints:**
1. **Inventory and Demand Fulfillment:**
   $$
   0 \leq x_i \leq \min\{I_i, d_i\}, \quad \forall i \in \mathcal{I}
   $$
   (Equivalently, $x_i \leq I_i$ and $x_i \leq d_i$ for all $i$.)

2. **Integrality:**
   $$
   x_i \in \mathbb{Z}_+, \quad \forall i \in \mathcal{I}
   $$

---

**Data Mapping:**  
- Table: `file_0_view_0` (from PizzaSalesDataset.csv)
    - Index set $\mathcal{I}$: All unique values in column "Product Name"
    - Parameter $A_i$: Column "Revenue"
    - Parameter $d_i$: Column "Demand"
    - Parameter $I_i$: Column "Initial Inventory"