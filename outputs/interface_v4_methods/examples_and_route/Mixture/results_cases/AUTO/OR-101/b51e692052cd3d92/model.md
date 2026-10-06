#### Abstract Mathematical Model

Let:
- $I$ = set of products, indexed by $i$ (from Product column in unit_product_profits.csv and device_time.csv)
- $K$ = set of devices, indexed by $k$ (from Device column in device_time.csv and monthly_device_capacity.csv)

Parameters:
- $p_i$ = unit profit of product $i$ (from file_2_view_0, column Unit_Profit)
- $a_{ki}$ = processing time required by product $i$ on device $k$ (from file_0_view_0, row Device $k$, column $i$)
- $c_k$ = monthly capacity of device $k$ (from file_1_view_0, column Monthly_Capacity)

Decision variables:
- $x_i \geq 0$ = production quantity of product $i$ (continuous)

Objective:
$$
\max \sum_{i \in I} p_i x_i
$$

Subject to (for all $k \in K$):
$$
\sum_{i \in I} a_{ki} x_i \leq c_k
$$

$$
x_i \geq 0 \quad \forall i \in I
$$

---

#### Data Mapping

- $I$ (products): file_2_view_0.Product, file_0_view_0 columns P1–P111
- $K$ (devices): file_0_view_0.Device, file_1_view_0.Device
- $p_i$: file_2_view_0.Unit_Profit, indexed by Product
- $a_{ki}$: file_0_view_0, row Device $k$, column $i$ (P1–P111)
- $c_k$: file_1_view_0.Monthly_Capacity, indexed by Device

All indices, parameters, and constraints are mapped directly from the supplied files and columns.