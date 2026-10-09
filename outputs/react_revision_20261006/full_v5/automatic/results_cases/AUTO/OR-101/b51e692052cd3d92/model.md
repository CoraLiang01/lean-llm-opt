## Mathematical Model

Let:
- $I$ = set of products, indexed by $i$ (from all Product in unit_product_profits.csv and device_time.csv)
- $K$ = set of devices, indexed by $k$ (from all Device in device_time.csv and monthly_device_capacity.csv)
- $x_i \geq 0$ = production quantity of product $i$ (continuous)

Parameters:
- $p_i$ = unit profit of product $i$ (from unit_product_profits.csv, column Unit_Profit, key Product)
- $a_{ki}$ = processing time required by product $i$ on device $k$ (from device_time.csv, row Device $k$, column $i$)
- $c_k$ = monthly operating capacity of device $k$ (from monthly_device_capacity.csv, column Monthly_Capacity, key Device)

### Objective
\[
\max \sum_{i \in I} p_i x_i
\]

### Constraints
\[
\sum_{i \in I} a_{ki} x_i \leq c_k \qquad \forall k \in K
\]
\[
x_i \geq 0 \qquad \forall i \in I
\]

---

## Data Mapping

- $I$ (products): All Product in file_2_view_0 (unit_product_profits.csv) and all columns (except Device) in file_0_view_0 (device_time.csv)
- $K$ (devices): All Device in file_1_view_0 (monthly_device_capacity.csv) and all Device in file_0_view_0 (device_time.csv)
- $p_i$: file_2_view_0, column Unit_Profit, key Product
- $a_{ki}$: file_0_view_0, row Device $k$, column $i$
- $c_k$: file_1_view_0, column Monthly_Capacity, key Device
- $x_i$: continuous, nonnegative, for all $i \in I$