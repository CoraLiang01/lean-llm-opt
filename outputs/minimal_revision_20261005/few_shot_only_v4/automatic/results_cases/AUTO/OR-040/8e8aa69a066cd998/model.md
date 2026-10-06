**Abstract Mathematical Model**

**Index Sets:**
- $I$: Set of areas (from products.csv, column ProductName)

**Parameters:**
- $p_i$: Benefit coefficient for area $i$ (from products.csv, column Value)
- $w_i$: Development unit weight for area $i$ (from products.csv, column Weight)
- $C$: Total development capacity (from capacity.csv, column Capacity)

**Decision Variables:**
- $x_i$: Integer, scale of development in area $i$ per day; $x_i \in \mathbb{Z}_{\geq 0}$

---

**Objective:**
\[
\max \sum_{i \in I} p_i x_i
\]

**Subject to:**
\[
\sum_{i \in I} w_i x_i \leq C
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

---

**Data Mapping**

- $I$: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA6/NYCPropertySales1/products.csv, column ProductName
- $p_i$: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA6/NYCPropertySales1/products.csv, column Value
- $w_i$: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA6/NYCPropertySales1/products.csv, column Weight
- $C$: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA6/NYCPropertySales1/capacity.csv, column Capacity