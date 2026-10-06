**Abstract Mathematical Model**

**Index Sets:**
- $I$: Set of products (from unit_product_profits.csv, column Product)
- $K$: Set of devices (from device_time.csv and monthly_device_capacity.csv, column Device)

**Parameters:**
- $p_i$: Unit profit of product $i$ (from unit_product_profits.csv, column Unit_Profit)
- $a_{ki}$: Processing time required by product $i$ on device $k$ (from device_time.csv, column $i$ for row Device $k$)
- $c_k$: Monthly operating capacity of device $k$ (from monthly_device_capacity.csv, column Monthly_Capacity)

**Decision Variables:**
- $x_i \geq 0$: Continuous quantity of product $i$ to produce

**Objective:**
\[
\max \sum_{i \in I} p_i x_i
\]

**Constraints:**
\[
\sum_{i \in I} a_{ki} x_i \leq c_k \qquad \forall k \in K
\]
\[
x_i \geq 0 \qquad \forall i \in I
\]

---

**Data Mapping**

- $I$ (products): /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture18/unit_product_profits.csv, column Product
- $K$ (devices): /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture18/device_time.csv, column Device
- $p_i$: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture18/unit_product_profits.csv, column Unit_Profit, keyed by Product
- $a_{ki}$: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture18/device_time.csv, column $i$ (P1–P111), row Device $k$
- $c_k$: /Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture18/monthly_device_capacity.csv, column Monthly_Capacity, keyed by Device

---

**Variable Domains:**  
$x_i \in \mathbb{R}_{\geq 0}$ (nonnegative continuous) for all $i \in I$