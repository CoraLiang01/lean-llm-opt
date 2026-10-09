##### Mathematical Model

Let  
- $I$ = set of platforms, indexed by $i$ (from PlatformId in capacity.csv)  
- $J$ = set of game genres, indexed by $j$ (from ProductName in products.csv)  
- $x_{ij}$ = number of units of genre $j$ listed on platform $i$ (decision variable, integer, $\geq 0$)  
- $v_j$ = value of one unit of genre $j$ (from Value in products.csv)  
- $w_j$ = memory requirement of one unit of genre $j$ (from Weight in products.csv)  
- $C_i$ = memory capacity of platform $i$ (from Capacity in capacity.csv)  

**Objective:**  
Maximize total value across all platforms:
$$
\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}
$$

**Subject to:**  
For each platform $i \in I$:
$$
\sum_{j \in J} w_j \, x_{ij} \leq C_i
$$

For all $i \in I$, $j \in J$:
$$
x_{ij} \in \mathbb{Z}_{\geq 0}
$$

---

##### Data Mapping

- $I$: PlatformId from /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/capacity.csv
- $C_i$: Capacity from /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/capacity.csv, keyed by PlatformId
- $J$: ProductName from /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/products.csv
- $v_j$: Value from /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/products.csv, keyed by ProductName
- $w_j$: Weight from /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA13/VideoGameSales1/products.csv, keyed by ProductName
- $x_{ij}$: Number of units of genre $j$ listed on platform $i$ (decision variable, integer, $\geq 0$)