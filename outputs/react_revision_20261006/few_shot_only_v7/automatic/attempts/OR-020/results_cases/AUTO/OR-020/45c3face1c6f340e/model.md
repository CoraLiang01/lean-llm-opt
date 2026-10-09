#### Symbolic Model

**Index Sets:**
- $I$: set of all products (from column ‘Product Name’)

**Parameters:**
- $A_i$: revenue per unit of product $i$ (from column ‘Revenue’)
- $d_i$: total demand for product $i$ (from column ‘Demand’)
- $I_i$: initial inventory for product $i$ (from column ‘Initial Inventory’)

**Decision Variables:**
- $x_i$: number of units of product $i$ to fulfill, $\forall i \in I$; $x_i \in \mathbb{Z}_+$

**Objective:**
\[
\max \sum_{i \in I} A_i x_i
\]

**Constraints:**
1. **Inventory constraint:** 
   \[
   x_i \leq I_i \quad \forall i \in I
   \]
2. **Demand constraint:** 
   \[
   x_i \leq d_i \quad \forall i \in I
   \]
3. **Nonnegativity and integrality:** 
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

#### Data Mapping

- Table: `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM11/SalesDatainBusinesses.csv`
    - Product set $I$: column `Product Name`
    - Revenue parameter $A_i$: column `Revenue`
    - Demand parameter $d_i$: column `Demand`
    - Initial inventory parameter $I_i$: column `Initial Inventory`