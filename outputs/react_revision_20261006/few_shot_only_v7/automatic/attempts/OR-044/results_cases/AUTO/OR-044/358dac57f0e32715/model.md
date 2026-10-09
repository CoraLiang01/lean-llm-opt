##### Mathematical Model

Let  
- $I$ = set of sections, indexed by $i$ (from capacity.csv, SectionID)  
- $J$ = set of products, indexed by $j$ (from products.csv, ProductName)  
- $x_{ij}$ = number of units of product $j$ to stock in section $i$ (integer, $\geq 0$)  
- $v_j$ = value (price) of product $j$  
- $w_j$ = shelf space requirement of product $j$  
- $C_i$ = display space capacity of section $i$  

**Objective:**  
$$
\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}
$$

**Subject to:**  
- Section capacity constraints (for all $i \in I$):  
$$
\sum_{j \in J} w_j \, x_{ij} \leq C_i
$$

- Integer and nonnegativity constraints (for all $i \in I$, $j \in J$):  
$$
x_{ij} \in \mathbb{Z}_{\geq 0}
$$

---

##### Data Mapping

- $I$: SectionID from `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA10/SalesOfASupermarket1/capacity.csv`
- $J$: ProductName from `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA10/SalesOfASupermarket1/products.csv`
- $C_i$: Capacity from `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA10/SalesOfASupermarket1/capacity.csv`, keyed by SectionID
- $v_j$: Value from `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA10/SalesOfASupermarket1/products.csv`, keyed by ProductName
- $w_j$: Weight from `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA10/SalesOfASupermarket1/products.csv`, keyed by ProductName
- $x_{ij}$: Number of units of product $j$ in section $i$ (decision variable, integer, $\geq 0$)