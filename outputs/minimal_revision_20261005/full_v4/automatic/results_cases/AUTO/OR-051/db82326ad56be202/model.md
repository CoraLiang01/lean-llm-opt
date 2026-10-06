**Abstract Mathematical Model**

**Index Sets:**
- $I$: set of cabinets, indexed by $i$ (from all CabinetID in file_0_view_0)
- $J$: set of coffee products, indexed by $j$ (from all ProductName in file_1_view_0)

**Parameters:**
- $c_i$: capacity of cabinet $i$ (from Capacity in file_0_view_0, indexed by CabinetID)
- $v_j$: value per unit of product $j$ (from Value in file_1_view_0, indexed by ProductName)
- $w_j$: weight per unit of product $j$ (from Weight in file_1_view_0, indexed by ProductName)

**Decision Variables:**
- $x_{ij}$: number of units of product $j$ to place in cabinet $i$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

**Objective:**
\[
\max \sum_{i \in I} \sum_{j \in J} v_j \, x_{ij}
\]

**Constraints:**
1. **Cabinet Capacity Constraints:**  
   For each cabinet $i \in I$,
   \[
   \sum_{j \in J} w_j \, x_{ij} \leq c_i
   \]
2. **Integrality and Nonnegativity:**  
   For all $i \in I$, $j \in J$,
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0}
   \]

---

**Data Mapping**

- $I$: All CabinetID in `file_0_view_0` (capacity.csv), column `CabinetID`
- $J$: All ProductName in `file_1_view_0` (products.csv), column `ProductName`
- $c_i$: `file_0_view_0`, columns `CabinetID`, `Capacity`
- $v_j$: `file_1_view_0`, columns `ProductName`, `Value`
- $w_j$: `file_1_view_0`, columns `ProductName`, `Weight`