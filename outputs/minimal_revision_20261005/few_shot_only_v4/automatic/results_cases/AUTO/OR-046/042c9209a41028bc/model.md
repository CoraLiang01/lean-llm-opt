**Mathematical Model**

**Index Sets:**
- Let $I$ be the set of products, indexed by $i$.

**Parameters:**
- $v_i$: Value (benefit) per unit of product $i$.
- $w_i$: Weight (stock space required) per unit of product $i$.
- $C$: Total stock capacity.

**Decision Variables:**
- $x_i \in \mathbb{Z}_{\geq 0}$: Number of units of product $i$ to order each day.

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

- $I$: All records in `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/products.csv`, column `ProductName`
- $v_i$: `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/products.csv`, column `Value`, keyed by `ProductName`
- $w_i$: `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/products.csv`, column `Weight`, keyed by `ProductName`
- $C$: `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/capacity.csv`, column `Capacity`