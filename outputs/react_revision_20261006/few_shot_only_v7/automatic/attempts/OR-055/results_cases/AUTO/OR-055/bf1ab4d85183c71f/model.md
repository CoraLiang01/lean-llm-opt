##### Sets
- $I$: set of display areas (indexed by $i$), from `capacity.csv` DisplayID
- $J$: set of boat types (indexed by $j$), from `products.csv` ProductName

##### Parameters
- $c_i$: capacity of display area $i$, from `capacity.csv` Capacity
- $v_j$: value of one unit of boat type $j$, from `products.csv` Value
- $w_j$: size (weight) of one unit of boat type $j$, from `products.csv` Weight

##### Decision Variables
- $x_{ij} \in \mathbb{Z}_{\geq 0}$: number of units of boat type $j$ placed in display area $i$

##### Objective
Maximize total value:
$$
\max \sum_{i \in I} \sum_{j \in J} v_j x_{ij}
$$

##### Constraints

1. **Display Area Capacity Constraints**  
For each display area $i \in I$:
$$
\sum_{j \in J} w_j x_{ij} \leq c_i
$$

2. **Nonnegativity and Integrality**
$$
x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J
$$

---

##### Data Mapping

- $I$: `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA21/BoatSales1/capacity.csv`, column `DisplayID`
- $J$: `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA21/BoatSales1/products.csv`, column `ProductName`
- $c_i$: `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA21/BoatSales1/capacity.csv`, column `Capacity`, keyed by `DisplayID`
- $v_j$: `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA21/BoatSales1/products.csv`, column `Value`, keyed by `ProductName`
- $w_j$: `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA21/BoatSales1/products.csv`, column `Weight`, keyed by `ProductName`