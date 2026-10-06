**Mathematical Model**

**Sets:**
- $S$: Set of shelves, indexed by $s$ (from capacity.csv, column ShelfID)
- $P$: Set of products, indexed by $p$ (from products.csv, column ProductName)

**Parameters:**
- $C_s$: Capacity of shelf $s$ (from capacity.csv, column Capacity)
- $v_p$: Value of product $p$ (from products.csv, column Value)
- $w_p$: Weight of product $p$ (from products.csv, column Weight)

**Decision Variables:**
- $x_{sp}$: Number of units of product $p$ placed on shelf $s$; $x_{sp} \in \mathbb{Z}_{\geq 0}$

---

**Objective:**
\[
\max \sum_{s \in S} \sum_{p \in P} v_p \cdot x_{sp}
\]

**Subject to:**

1. **Shelf Capacity Constraints:**
   \[
   \sum_{p \in P} w_p \cdot x_{sp} \leq C_s \qquad \forall s \in S
   \]

2. **Integrality and Nonnegativity:**
   \[
   x_{sp} \in \mathbb{Z}_{\geq 0} \qquad \forall s \in S,\, p \in P
   \]

---

**Data Mapping**

| Symbol         | Source Table                                                                                                   | Column         | Key(s)                |
|----------------|--------------------------------------------------------------------------------------------------------------|----------------|-----------------------|
| $S$            | /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA19/BigMartSalesData1/capacity.csv | ShelfID        |                       |
| $P$            | /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA19/BigMartSalesData1/products.csv | ProductName    |                       |
| $C_s$          | /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA19/BigMartSalesData1/capacity.csv | Capacity       | ShelfID               |
| $v_p$          | /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA19/BigMartSalesData1/products.csv | Value          | ProductName           |
| $w_p$          | /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA19/BigMartSalesData1/products.csv | Weight         | ProductName           |
| $x_{sp}$       | (decision variable)                                                                                           |                | ShelfID, ProductName  |