#### Symbolic Model

**Index Sets:**
- $I$ : set of all products with identifier ‘id999’

**Parameters:**
- $A_i$ : revenue per unit of product $i \in I$ (from column ‘Revenue’)
- $d_i$ : deterministic demand for product $i \in I$ (from column ‘Demand’)
- $I_i$ : initial inventory for product $i \in I$ (from column ‘Initial Inventory’)

**Decision Variables:**
- $x_i$ : number of units of product $i \in I$ to fulfill, $x_i \in \mathbb{Z}_+, \forall i \in I$

**Objective:**
\[
\max \sum_{i \in I} A_i x_i
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
3. **Non-negativity and Integrality:**
   \[
   x_i \in \mathbb{Z}_+, \quad \forall i \in I
   \]

---

#### Data Mapping

- Table: `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM2/OnlineRetailSalesDataset.csv`
    - Index set $I$: all records where `id_number` = ‘id999’
    - Parameter $A_i$: column `Revenue`
    - Parameter $d_i$: column `Demand`
    - Parameter $I_i$: column `Initial Inventory`