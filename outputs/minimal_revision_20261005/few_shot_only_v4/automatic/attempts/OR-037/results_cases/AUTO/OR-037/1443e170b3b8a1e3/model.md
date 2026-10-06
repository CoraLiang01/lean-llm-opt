**Abstract Mathematical Model**

**Index Sets:**
- $I$: Set of vehicle types (from products.csv, column ProductName)

**Parameters:**
- $v_i$: Profit per unit of vehicle $i$ (from products.csv, column Value)
- $w_i$: Inventory space required per unit of vehicle $i$ (from products.csv, column Weight)
- $C$: Total inventory capacity (from capacity.csv, column Capacity)

**Decision Variables:**
- $x_i \in \mathbb{Z}_{\geq 0}$: Number of vehicles of type $i$ to order per day

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

- $I$: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/products.csv, column ProductName
- $v_i$: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/products.csv, column Value
- $w_i$: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/products.csv, column Weight
- $C$: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/capacity.csv, column Capacity