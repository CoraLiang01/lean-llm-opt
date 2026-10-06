**Abstract Mathematical Model**

**Sets:**
- $I$: Set of drug products (indexed by $i$).

**Parameters:**
- $v_i$: Value (benefit) of one unit of drug $i$.
- $w_i$: Weight (stock space required) for one unit of drug $i$.
- $C$: Total stock capacity for the pharmacy chain.

**Decision Variables:**
- $x_i \in \mathbb{Z}_{\geq 0}$: Number of units of drug $i$ to order each day.

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

**Data Mapping**

- $I$: All records in `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA9/PharmaSalesData2/products.csv`, column `ProductName`
- $v_i$: `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA9/PharmaSalesData2/products.csv`, column `Value`
- $w_i$: `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA9/PharmaSalesData2/products.csv`, column `Weight`
- $C$: `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA9/PharmaSalesData2/capacity.csv`, column `Capacity`