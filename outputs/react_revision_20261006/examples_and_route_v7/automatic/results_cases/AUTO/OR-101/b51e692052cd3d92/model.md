#### Mathematical Model

Let:
- $I$ = set of products, indexed by $i$ (from all Product in unit_product_profits.csv and device_time.csv)
- $K$ = set of devices, indexed by $k$ (from all Device in device_time.csv and monthly_device_capacity.csv)
- $x_i \geq 0$ = production quantity of product $i$ (continuous)

Parameters:
- $p_i$ = unit profit of product $i$ (from unit_product_profits.csv, column Unit_Profit, key Product)
- $a_{ki}$ = processing time required by product $i$ on device $k$ (from device_time.csv, row Device $k$, column $i$)
- $c_k$ = monthly capacity of device $k$ (from monthly_device_capacity.csv, column Monthly_Capacity, key Device)

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

#### Data Mapping

- $I$ (products): All Product in unit_product_profits.csv (table_id: file_2_view_0, column: Product) and device_time.csv (table_id: file_0_view_0, columns: P1...P111)
- $K$ (devices): All Device in device_time.csv (table_id: file_0_view_0, column: Device) and monthly_device_capacity.csv (table_id: file_1_view_0, column: Device)
- $p_i$: file_2_view_0, columns Product, Unit_Profit (key: Product)
- $a_{ki}$: file_0_view_0, row Device $k$, column $i$ (columns: P1...P111, key: Device)
- $c_k$: file_1_view_0, columns Device, Monthly_Capacity (key: Device)
- $x_i$: continuous, nonnegative, for all $i \in I$