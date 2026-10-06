**Abstract Mathematical Model**

**Index Sets:**
- $C$: Set of cabinets, indexed by $i$ (from capacity.csv, column CabinetID)
- $P$: Set of coffee products, indexed by $j$ (from products.csv, column ProductName)

**Parameters:**
- $v_j$: Value per unit of product $j$ (from products.csv, column Value)
- $w_j$: Weight per unit of product $j$ (from products.csv, column Weight)
- $cap_i$: Capacity of cabinet $i$ (from capacity.csv, column Capacity)

**Decision Variables:**
- $x_{ij}$: Number of units of product $j$ to place in cabinet $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

**Objective:**
\[
\max \sum_{i \in C} \sum_{j \in P} v_j \cdot x_{ij}
\]

**Constraints:**
1. **Cabinet Capacity Constraints:**  
   For each cabinet $i \in C$,
   \[
   \sum_{j \in P} w_j \cdot x_{ij} \leq cap_i
   \]
2. **Integrality and Nonnegativity:**
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in C,\, j \in P
   \]

---

**Data Mapping**

- $C$: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA17/CoffeeChainSalesAnalysis1/capacity.csv, column CabinetID
- $cap_i$: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA17/CoffeeChainSalesAnalysis1/capacity.csv, column Capacity
- $P$: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA17/CoffeeChainSalesAnalysis1/products.csv, column ProductName
- $v_j$: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA17/CoffeeChainSalesAnalysis1/products.csv, column Value
- $w_j$: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA17/CoffeeChainSalesAnalysis1/products.csv, column Weight