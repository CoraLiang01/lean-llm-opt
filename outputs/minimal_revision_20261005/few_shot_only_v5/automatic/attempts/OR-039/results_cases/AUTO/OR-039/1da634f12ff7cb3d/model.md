**Abstract Mathematical Model**

**Index Sets:**
- $I$: Set of vehicle types (from products.csv, column ProductName)
- $J$: Set of warehouses (from capacity.csv, column Warehouse ID)

**Parameters:**
- $v_i$: Value (benefit coefficient) of vehicle type $i$ (from products.csv, column Value)
- $w_i$: Weight (space requirement) of vehicle type $i$ (from products.csv, column Weight)
- $C_j$: Capacity of warehouse $j$ (from capacity.csv, column Capacity)

**Decision Variables:**
- $x_{ij}$: Number of units of vehicle type $i$ to store in warehouse $j$; $x_{ij} \in \mathbb{Z}_{\geq 0}$

**Objective:**
\[
\max \sum_{j \in J} \sum_{i \in I} v_i \cdot x_{ij}
\]

**Constraints:**
1. **Warehouse Capacity Constraints:**  
   For each warehouse $j \in J$,
   \[
   \sum_{i \in I} w_i \cdot x_{ij} \leq C_j
   \]
2. **Non-negativity and Integrality:**
   \[
   x_{ij} \in \mathbb{Z}_{\geq 0} \quad \forall i \in I,\, j \in J
   \]

---

**Data Mapping**

- $I$ (vehicle types): /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA5/NewCarSalesInNorway2/products.csv, column ProductName
- $v_i$: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA5/NewCarSalesInNorway2/products.csv, column Value
- $w_i$: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA5/NewCarSalesInNorway2/products.csv, column Weight
- $J$ (warehouses): /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA5/NewCarSalesInNorway2/capacity.csv, column Warehouse ID
- $C_j$: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA5/NewCarSalesInNorway2/capacity.csv, column Capacity