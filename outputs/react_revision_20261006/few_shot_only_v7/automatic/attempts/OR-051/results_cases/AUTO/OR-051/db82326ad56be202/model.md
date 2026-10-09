##### Mathematical Model

Let  
- $I$ = set of cabinets (indexed by $i$), from the CabinetID column of capacity.csv  
- $J$ = set of coffee products (indexed by $j$), from the ProductName column of products.csv  

Parameters:  
- $c_i$ = capacity of cabinet $i$ (from capacity.csv, Capacity)  
- $v_j$ = value per unit of product $j$ (from products.csv, Value)  
- $w_j$ = weight per unit of product $j$ (from products.csv, Weight)  

Decision variables:  
- $x_{ij}$ = number of units of product $j$ to place in cabinet $i$  
  ($x_{ij} \in \mathbb{Z}_{\geq 0}$, for all $i \in I$, $j \in J$)

Objective:  
$\displaystyle \max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}$

Subject to:  
$\displaystyle \sum_{j \in J} w_j x_{ij} \leq c_i \quad \forall i \in I$  
$x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J$

---

##### Data Mapping

- $I$: CabinetID from /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA17/CoffeeChainSalesAnalysis1/capacity.csv
- $c_i$: Capacity from /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA17/CoffeeChainSalesAnalysis1/capacity.csv, keyed by CabinetID
- $J$: ProductName from /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA17/CoffeeChainSalesAnalysis1/products.csv
- $v_j$: Value from /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA17/CoffeeChainSalesAnalysis1/products.csv, keyed by ProductName
- $w_j$: Weight from /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA17/CoffeeChainSalesAnalysis1/products.csv, keyed by ProductName