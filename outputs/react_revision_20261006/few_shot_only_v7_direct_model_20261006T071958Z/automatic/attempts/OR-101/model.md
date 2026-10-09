**Mathematical Model**

Let:
- $I$ = set of products, indexed by $i$ (from all Product in file_2_view_0)
- $D$ = set of devices, indexed by $d$ (from all Device in file_0_view_0 and file_1_view_0)
- $x_i$ = quantity of product $i$ to produce (continuous, $x_i \geq 0$)

Parameters:
- $p_i$ = unit profit of product $i$ (from file_2_view_0, column Unit_Profit)
- $t_{d,i}$ = processing time required on device $d$ per unit of product $i$ (from file_0_view_0, column $Pj$ for product $Pj$ on device $d$)
- $c_d$ = monthly capacity of device $d$ (from file_1_view_0, column Monthly_Capacity)

**Objective:**
\[
\max \sum_{i \in I} p_i x_i
\]

**Constraints:**
\[
\sum_{i \in I} t_{d,i} x_i \leq c_d \qquad \forall d \in D
\]
\[
x_i \geq 0 \qquad \forall i \in I
\]

**Data Mapping**

- $I$: All Product in file_2_view_0["Product"]
- $D$: All Device in file_0_view_0["Device"] and file_1_view_0["Device"]
- $p_i$: file_2_view_0["Unit_Profit"], keyed by Product
- $t_{d,i}$: file_0_view_0, row Device $d$, column $Pj$ for product $Pj$
- $c_d$: file_1_view_0["Monthly_Capacity"], keyed by Device

- Decision variables: $x_i$ (continuous, nonnegative), for all $i \in I$