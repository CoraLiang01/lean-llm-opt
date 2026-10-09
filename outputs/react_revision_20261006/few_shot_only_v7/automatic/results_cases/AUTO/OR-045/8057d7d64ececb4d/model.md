##### Mathematical Model

Let $I$ be the set of produce types indexed by $i$ (from the ProductName column in products.csv).

**Parameters:**
- $v_i$: Value (benefit) per unit of produce $i$ (from products.csv, Value)
- $w_i$: Weight per unit of produce $i$ (from products.csv, Weight)
- $C$: Total inventory capacity (from capacity.csv, Capacity)

**Decision Variables:**
- $x_i \in \mathbb{Z}_{\geq 0}$: Number of units of produce $i$ to order daily

**Objective:**
\[
\max \sum_{i \in I} v_i x_i
\]

**Subject to:**
\[
\sum_{i \in I} w_i x_i \leq C
\]
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i \in I
\]

---

##### Data Mapping

- $I$: All records in /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv, column ProductName
- $v_i$: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv, column Value, keyed by ProductName
- $w_i$: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv, column Weight, keyed by ProductName
- $C$: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/capacity.csv, column Capacity