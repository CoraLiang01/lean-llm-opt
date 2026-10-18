#### Abstract Mathematical Model

**Index Set:**
- $I$ : set of all car models classified as ‘FDK57’ (indexed by $i$)

**Parameters:**
- $A_i$ : revenue per unit for car model $i$ (from column ‘Revenue’)
- $d_i$ : total demand for car model $i$ (from column ‘Demand’)
- $I_i$ : initial inventory for car model $i$ (from column ‘Initial Inventory’)

**Decision Variables:**
- $x_i$ : number of units of car model $i$ to fulfill, $x_i \in \mathbb{Z}_+, \forall i \in I$

**Objective:**
\[
\max \sum_{i \in I} A_i \cdot x_i
\]

**Constraints:**
1. **Inventory Constraint:** 
   \[
   x_i \leq I_i, \quad \forall i \in I
   \]
2. **Demand Constraint:** 
   \[
   x_i \leq d_i, \quad \forall i \in I
   \]
3. **Nonnegativity and Integrality:**
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

**Data Mapping:**

- Source Table: `file_0_view_0` (from `BigMartSales.csv`)
- Index Set $I$: All records where `Product Name` = ‘FDK57’
- Parameter $A_i$: column `Revenue`
- Parameter $d_i$: column `Demand`
- Parameter $I_i$: column `Initial Inventory`