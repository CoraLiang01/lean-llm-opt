##### Abstract Mathematical Model

**Index Set:**  
Let $I$ be the set of all products identified by the values in the column `Full_Product_Name`.

**Parameters:**  
For each $i \in I$:
- $A_i$: revenue per unit of product $i$ (from column `Revenue`)
- $d_i$: total demand for product $i$ (from column `Demand`)
- $I_i$: initial inventory for product $i$ (from column `Initial Inventory`)

**Decision Variables:**  
For each $i \in I$:
- $x_i$: number of units of product $i$ to fulfill  
  Domain: $x_i \in \mathbb{Z}_+, \quad 0 \leq x_i \leq \min\{d_i, I_i\}$

**Objective:**  
$\max \sum_{i \in I} A_i \cdot x_i$

**Constraints:**  
For all $i \in I$:
- $x_i \leq d_i$  (Demand constraint)
- $x_i \leq I_i$  (Inventory constraint)
- $x_i \geq 0$ and integer

##### Data Mapping

- Table: `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM22/DairyGoodsSalesDataset.csv`, table_id: `file_0_view_0`
    - Index set $I$: all unique values in column `Full_Product_Name`
    - Parameter $A_i$: column `Revenue`
    - Parameter $d_i$: column `Demand`
    - Parameter $I_i$: column `Initial Inventory`