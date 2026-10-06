**Abstract Mathematical Model**

**Index Sets:**
- $I$: Set of display areas, indexed by $i$ (from capacity.csv, column DisplayID)
- $J$: Set of boat types, indexed by $j$ (from products.csv, column ProductName)

**Parameters:**
- $c_i$: Capacity of display area $i$ (capacity.csv, column Capacity)
- $v_j$: Value of one unit of boat type $j$ (products.csv, column Value)
- $w_j$: Size (weight) of one unit of boat type $j$ (products.csv, column Weight)

**Decision Variables:**
- $x_{ij}$: Number of units of boat type $j$ to place in display area $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

---

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}
\]

**Subject to:**

1. **Display Area Capacity Constraints:**
   \[
   \sum_{j \in J} w_j \, x_{ij} \leq c_i \qquad \forall i \in I
   \]

2. **Integrality and Nonnegativity:**
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
   \]

---

**Data Mapping**

- $I$ (Display areas): `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA21/BoatSales1/capacity.csv`, column `DisplayID`
- $c_i$: `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA21/BoatSales1/capacity.csv`, column `Capacity`
- $J$ (Boat types): `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA21/BoatSales1/products.csv`, column `ProductName`
- $v_j$: `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA21/BoatSales1/products.csv`, column `Value`
- $w_j$: `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA21/BoatSales1/products.csv`, column `Weight`