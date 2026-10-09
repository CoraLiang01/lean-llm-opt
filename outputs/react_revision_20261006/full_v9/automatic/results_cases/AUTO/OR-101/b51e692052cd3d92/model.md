### Mathematical Model

Let:
- $I$ = set of products, indexed by $i$ (from all Product in unit_product_profits.csv and device_time.csv)
- $K$ = set of devices, indexed by $k$ (from all Device in device_time.csv and monthly_device_capacity.csv)
- $x_i \geq 0$ = production quantity of product $i$ (continuous)

Parameters:
- $p_i$ = unit profit of product $i$
- $a_{ki}$ = processing time required by product $i$ on device $k$
- $c_k$ = monthly operating capacity of device $k$

#### Objective:
$$
\max \sum_{i \in I} p_i x_i
$$

#### Constraints:
For each device $k \in K$:
$$
\sum_{i \in I} a_{ki} x_i \leq c_k
$$

For each product $i \in I$:
$$
x_i \geq 0
$$

---

### Data Mapping

- $I$ (products): All Product in file_2_view_0 (unit_product_profits.csv) and columns of file_0_view_0 (device_time.csv) except "Device"
- $K$ (devices): All Device in file_0_view_0 (device_time.csv) and file_1_view_0 (monthly_device_capacity.csv)
- $p_i$: file_2_view_0, column Unit_Profit, key Product
- $a_{ki}$: file_0_view_0, value at row Device $k$, column $i$
- $c_k$: file_1_view_0, column Monthly_Capacity, key Device
- $x_i$: continuous, nonnegative, for each $i \in I$