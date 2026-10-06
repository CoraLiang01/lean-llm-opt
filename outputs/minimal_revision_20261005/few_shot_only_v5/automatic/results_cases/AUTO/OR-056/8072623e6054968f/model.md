**Abstract Mathematical Model**

**Sets:**
- $I$: Set of display areas, indexed by $i$ (from capacity.csv, column DisplayID)
- $J$: Set of vessel types, indexed by $j$ (from products.csv, column ProductName)

**Parameters:**
- $c_i$: Capacity of display area $i$ (capacity.csv, column Capacity)
- $v_j$: Value of vessel type $j$ (products.csv, column Value)
- $w_j$: Size (weight) of vessel type $j$ (products.csv, column Weight)

**Decision Variables:**
- $x_{ij} \in \mathbb{Z}_{\geq 0}$: Number of vessels of type $j$ to place in display area $i$

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

2. **Nonnegativity and Integrality:**
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0} \qquad \forall i \in I,\, j \in J
   \]

---

**Data Mapping**

- $I$ (Display Areas): `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA22/BoatSales2/capacity.csv`, column `DisplayID`
- $c_i$: `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA22/BoatSales2/capacity.csv`, column `Capacity`
- $J$ (Vessel Types): `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA22/BoatSales2/products.csv`, column `ProductName`
- $v_j$: `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA22/BoatSales2/products.csv`, column `Value`
- $w_j$: `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA22/BoatSales2/products.csv`, column `Weight`