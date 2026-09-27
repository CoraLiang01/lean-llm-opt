#### Abstract Mathematical Optimization Model

**Index Sets:**
- $I$ : set of all baked goods (indexed by $i$)

**Parameters:**
- $A_i$ : revenue per unit of baked good $i$ (from column ‘Revenue’)
- $d_i$ : total demand for baked good $i$ (from column ‘Demand’)
- $I_i$ : initial inventory for baked good $i$ (from column ‘Initial Inventory’)

**Decision Variables:**
- $x_i$ : quantity of baked good $i$ to fulfill, for all $i \in I$

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
3. Non-negativity:
   \[
   x_i \geq 0 \qquad \forall i \in I
   \]

---

#### Data Mapping

- Table: `Frenchbakerydailysales.csv` (table_id: `file_0_view_0`)
    - Baked good identifier: `Product Name`
    - Revenue per unit: `Revenue`
    - Initial inventory: `Initial Inventory`
    - Demand: `Demand`