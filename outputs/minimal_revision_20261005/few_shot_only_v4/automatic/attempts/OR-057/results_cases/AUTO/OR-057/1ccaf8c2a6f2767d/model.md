**Abstract Mathematical Model**

**Index Sets:**
- $I$: Set of platforms (indexed by $i$), from capacity.csv [PlatformID]
- $J$: Set of games (indexed by $j$), from products.csv [ProductName]

**Parameters:**
- $c_i$: Capacity of platform $i$ (from capacity.csv [Capacity])
- $v_j$: Value of game $j$ (from products.csv [Value])
- $w_j$: Memory requirement of game $j$ (from products.csv [Weight])

**Decision Variables:**
- $x_{ij}$: Number of units of game $j$ to list on platform $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \cdot x_{ij}
\]

**Constraints:**
1. **Platform Capacity Constraints:**  
   For each platform $i \in I$,
   \[
   \sum_{j \in J} w_j \cdot x_{ij} \leq c_i
   \]
2. **Integrality and Nonnegativity:**
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J
   \]

---

**Data Mapping**

- $I$ (platforms): /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA16/VideoGameSalesWithRatings1/capacity.csv, column [PlatformID]
- $c_i$: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA16/VideoGameSalesWithRatings1/capacity.csv, column [Capacity]
- $J$ (games): /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA16/VideoGameSalesWithRatings1/products.csv, column [ProductName]
- $v_j$: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA16/VideoGameSalesWithRatings1/products.csv, column [Value]
- $w_j$: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA16/VideoGameSalesWithRatings1/products.csv, column [Weight]