##### Mathematical Model

Let  
- $I$ = set of display areas, indexed by $i$ (DisplayID from capacity.csv)  
- $J$ = set of boat types, indexed by $j$ (ProductName from products.csv)  
- $x_{ij}$ = number of vessels of type $j$ placed in display area $i$ (decision variable, integer, $\geq 0$)  
- $v_j$ = value of vessel type $j$  
- $w_j$ = weight (size) of vessel type $j$  
- $C_i$ = capacity of display area $i$  

**Objective:**  
Maximize total value of all vessels displayed:
$$
\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}
$$

**Subject to:**  
For each display area $i \in I$:
$$
\sum_{j \in J} w_j \, x_{ij} \leq C_i
$$

For all $i \in I$, $j \in J$:
$$
x_{ij} \in \mathbb{Z}_{\geq 0}
$$

---

##### Data Mapping

- $I$: DisplayID from `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA22/BoatSales2/capacity.csv`
- $C_i$: Capacity from `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA22/BoatSales2/capacity.csv`, keyed by DisplayID
- $J$: ProductName from `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA22/BoatSales2/products.csv`
- $v_j$: Value from `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA22/BoatSales2/products.csv`, keyed by ProductName
- $w_j$: Weight from `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA22/BoatSales2/products.csv`, keyed by ProductName
- $x_{ij}$: Number of vessels of type $j$ in display area $i$ (decision variable, integer, $\geq 0$)