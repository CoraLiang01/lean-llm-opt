#### Symbolic Model

**Index Sets:**
- $F$: set of all products with 'Fashion' in their 'Product Name' (from the source table).

**Parameters:**
- $A_i$: revenue per unit of product $i \in F$ (from column 'Revenue').
- $d_i$: total demand for product $i \in F$ (from column 'Demand').
- $I_i$: initial inventory for product $i \in F$ (from column 'Initial Inventory').

**Decision Variables:**
- $x_i$: number of units of product $i \in F$ to fulfill, $x_i \in \mathbb{Z}_+$.

**Objective:**
\[
\max \sum_{i \in F} A_i x_i
\]

**Constraints:**
1. **Inventory Bound:** 
   \[
   x_i \leq I_i \quad \forall i \in F
   \]
2. **Demand Bound:** 
   \[
   x_i \leq d_i \quad \forall i \in F
   \]
3. **Nonnegativity and Integrality:** 
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in F
   \]

#### Data Mapping

- Table: `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM17/SupermarketSales.csv`
    - Index set $F$: rows where `Product Name` contains 'Fashion'
    - $A_i$: column `Revenue`
    - $d_i$: column `Demand`
    - $I_i$: column `Initial Inventory`