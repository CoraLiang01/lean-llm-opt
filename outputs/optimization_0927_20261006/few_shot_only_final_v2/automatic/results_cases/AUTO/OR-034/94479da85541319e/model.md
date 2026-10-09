---

### Sets

- $I$: Set of all products, indexed by $i$.

### Parameters

- $A_i$: Revenue per unit of product $i$.  
- $d_i$: Demand for product $i$.  
- $I_i$: Initial Inventory for product $i$.

### Decision Variables

- $x_i$: Quantity of product $i$ to fulfill, $\forall i \in I$.

### Objective

$$
\max \sum_{i \in I} A_i \cdot x_i
$$

### Constraints

1. **Demand and Inventory Fulfillment Bounds**  
   $$
   0 \leq x_i \leq \min\{d_i, I_i\}, \quad \forall i \in I
   $$

   (Or, equivalently, two separate constraints:)
   $$
   x_i \leq d_i, \quad \forall i \in I
   $$
   $$
   x_i \leq I_i, \quad \forall i \in I
   $$
   $$
   x_i \geq 0, \quad \forall i \in I
   $$

### Data Mapping

- **Table:** `Frenchbakerydailysales.csv` (table_id: `file_0_view_0`)
    - **Product Name** $\rightarrow$ Index set $I$
    - **Revenue** $\rightarrow$ Parameter $A_i$
    - **Demand** $\rightarrow$ Parameter $d_i$
    - **Initial Inventory** $\rightarrow$ Parameter $I_i$

---