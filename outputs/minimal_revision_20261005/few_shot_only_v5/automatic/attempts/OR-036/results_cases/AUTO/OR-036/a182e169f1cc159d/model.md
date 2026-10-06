**Abstract Mathematical Model**

**Index Sets:**
- $I$: Set of vehicle types (from products.csv, column ProductName)

**Parameters:**
- $v_i$: Value (benefit) of vehicle type $i$ (from products.csv, column Value)
- $w_i$: Weight (inventory space required) of vehicle type $i$ (from products.csv, column Weight)
- $C$: Total inventory capacity (from capacity.csv, column Capacity)

**Decision Variables:**
- $x_i$: Number of units of vehicle type $i$ to order daily; $x_i \in \mathbb{Z}_{\geq 0}$

**Objective:**
\[
\max \sum_{i \in I} v_i x_i
\]

**Constraints:**
\[
\sum_{i \in I} w_i x_i \leq C
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

---

**Data Mapping**

- $I$: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA2/CarSales1/products.csv, column ProductName
- $v_i$: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA2/CarSales1/products.csv, column Value
- $w_i$: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA2/CarSales1/products.csv, column Weight
- $C$: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA2/CarSales1/capacity.csv, column Capacity