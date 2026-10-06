**Abstract Mathematical Model**

**Index Sets:**
- $I$: Set of bread types (from products.csv, column ProductName)

**Parameters:**
- $p_i$: Expected profit per unit of bread type $i$ (products.csv, column Value)
- $w_i$: Storage weight per unit of bread type $i$ (products.csv, column Weight)
- $C$: Total storage capacity (capacity.csv, column Capacity)

**Decision Variables:**
- $x_i$: Number of units of bread type $i$ to order each day; $x_i \in \mathbb{Z}_{\geq 0}$

**Objective:**
\[
\max \sum_{i \in I} p_i x_i
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

- $I$: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv, column ProductName
- $p_i$: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv, column Value
- $w_i$: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv, column Weight
- $C$: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/capacity.csv, column Capacity