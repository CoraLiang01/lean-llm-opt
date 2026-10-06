#### Abstract Mathematical Optimization Model

**Index Set:**
- $I$ : set of all baked goods (indexed by $i$)

**Parameters:**
- $A_i$ : revenue per unit of baked good $i$ (from column ‘Revenue’)
- $d_i$ : total demand for baked good $i$ (from column ‘Demand’)
- $I_i$ : initial inventory for baked good $i$ (from column ‘Initial Inventory’)

**Decision Variables:**
- $x_i$ : quantity of baked good $i$ to fulfill, $\forall i \in I$

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. Inventory constraint:
   \[
   x_i \leq I_i \qquad \forall i \in I
   \]
2. Demand constraint:
   \[
   x_i \leq d_i \qquad \forall i \in I
   \]
3. Non-negativity and integrality:
   \[
   x_i \geq 0,\quad x_i \in \mathbb{Z} \qquad \forall i \in I
   \]

---

**Data Mapping:**

- Table: `Frenchbakerydailysales.csv` (table_id: `file_0_view_0`)
    - Index set $I$: all rows, column `Product Name`
    - Parameter $A_i$: column `Revenue`
    - Parameter $d_i$: column `Demand`
    - Parameter $I_i$: column `Initial Inventory`