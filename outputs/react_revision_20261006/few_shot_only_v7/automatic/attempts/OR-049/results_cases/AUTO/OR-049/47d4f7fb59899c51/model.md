#### Mathematical Model

**Sets:**
- $S$: set of shelves (indexed by $i$), from `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA15/RetailSalesAnalysis1/capacity.csv`, column `ShelfID`
- $P$: set of products (indexed by $j$), from `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA15/RetailSalesAnalysis1/products.csv`, column `ProductName`

**Parameters:**
- $c_i$: capacity of shelf $i$ (from `capacity.csv`, column `Capacity`)
- $v_j$: value per unit of product $j$ (from `products.csv`, column `Value`)
- $w_j$: weight per unit of product $j$ (from `products.csv`, column `Weight`)

**Decision Variables:**
- $x_{ij} \in \mathbb{Z}_{\geq 0}$: number of units of product $j$ placed on shelf $i$

**Objective:**
\[
\max \sum_{i \in S} \sum_{j \in P} v_j \, x_{ij}
\]

**Constraints:**
1. **Shelf Capacity Constraints:**  
   For each shelf $i \in S$,
   \[
   \sum_{j \in P} w_j \, x_{ij} \leq c_i
   \]
2. **Integrality and Nonnegativity:**  
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in S,\, j \in P
   \]

---

#### Data Mapping

- $S$ (shelves): `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA15/RetailSalesAnalysis1/capacity.csv`, column `ShelfID`
- $c_i$: `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA15/RetailSalesAnalysis1/capacity.csv`, column `Capacity`
- $P$ (products): `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA15/RetailSalesAnalysis1/products.csv`, column `ProductName`
- $v_j$: `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA15/RetailSalesAnalysis1/products.csv`, column `Value`
- $w_j$: `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA15/RetailSalesAnalysis1/products.csv`, column `Weight`