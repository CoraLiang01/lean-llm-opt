### Mathematical Model

**Sets:**
- $P$: set of products (Product in file_2_view_0 and columns in file_0_view_0 except "Device")
- $D$: set of devices (Device in file_1_view_0 and file_0_view_0)

**Parameters:**
- $c_p$: unit profit of product $p$ (Unit_Profit, file_2_view_0, Product $p$)
- $a_{dp}$: processing time required by product $p$ on device $d$ (file_0_view_0, Device $d$, column $p$)
- $b_d$: monthly capacity of device $d$ (Monthly_Capacity, file_1_view_0, Device $d$)

**Decision Variables:**
- $x_p \geq 0$: production quantity of product $p$ (continuous)

**Objective:**
\[
\max \sum_{p \in P} c_p x_p
\]

**Constraints:**
\[
\sum_{p \in P} a_{dp} x_p \leq b_d \qquad \forall d \in D
\]
\[
x_p \geq 0 \qquad \forall p \in P
\]

---

### Data Mapping

- $P$: All products with Product in file_2_view_0 and as columns (except "Device") in file_0_view_0
- $D$: All devices with Device in file_1_view_0 and file_0_view_0
- $c_p$: file_2_view_0, column Unit_Profit, key Product $p$
- $a_{dp}$: file_0_view_0, row Device $d$, column $p$
- $b_d$: file_1_view_0, column Monthly_Capacity, key Device $d$
- $x_p$: continuous, nonnegative, for each $p \in P$