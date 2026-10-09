ABSTRACT MATHEMATICAL MODEL

Index Sets:
- $P$: set of products (Product in file_2_view_0, columns Product, Unit_Profit)
- $D$: set of devices (Device in file_1_view_0, columns Device, Monthly_Capacity)

Parameters:
- $u_p$: unit profit of product $p$ (file_2_view_0, column Unit_Profit, key Product)
- $t_{d,p}$: processing time required by product $p$ on device $d$ (file_0_view_0, row Device, column $p$)
- $c_d$: monthly operating capacity of device $d$ (file_1_view_0, column Monthly_Capacity, key Device)

Decision Variables:
- $x_p \geq 0$: continuous production quantity of product $p$ to produce in the month

Objective:
\[
\max \sum_{p \in P} u_p x_p
\]

Subject to (for all $d \in D$):
\[
\sum_{p \in P} t_{d,p} x_p \leq c_d
\]
\[
x_p \geq 0 \quad \forall p \in P
\]

Data Mapping:
- $P$: file_2_view_0, column Product
- $D$: file_1_view_0, column Device
- $u_p$: file_2_view_0, column Unit_Profit, key Product
- $t_{d,p}$: file_0_view_0, row Device, column $p$
- $c_d$: file_1_view_0, column Monthly_Capacity, key Device
- $x_p$: continuous, nonnegative, indexed by $p \in P$