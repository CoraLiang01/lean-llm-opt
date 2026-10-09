#### Abstract Mathematical Optimization Model

**Index Set:**
- $I$: set of all products classified as ‘27in’ (from column ‘Product Name’).

**Parameters:**
- $A_i$: revenue per unit of product $i \in I$ (from column ‘Revenue’).
- $d_i$: deterministic demand for product $i \in I$ (from column ‘Demand’).
- $I_i$: initial inventory for product $i \in I$ (from column ‘Initial Inventory’).

**Decision Variables:**
- $x_i$: number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+, \forall i \in I$.

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. Inventory and Demand Fulfillment:
   \[
   0 \leq x_i \leq \min\{d_i, I_i\}, \quad \forall i \in I
   \]
   (Equivalently, $x_i \leq d_i$ and $x_i \leq I_i$ for all $i$.)

2. Integer Variables:
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

**Data Mapping:**

- Table ID: file_0_view_0 (from Salesorders.csv)
    - Product Name: identifies set $I$
    - Revenue: parameter $A_i$
    - Demand: parameter $d_i$
    - Initial Inventory: parameter $I_i$