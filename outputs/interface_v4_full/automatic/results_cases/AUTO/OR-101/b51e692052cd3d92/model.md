## Abstract Mathematical Model

Let:
- $I$ = set of products, indexed by $i$ (from Product column in unit_product_profits.csv and device_time.csv)
- $K$ = set of devices, indexed by $k$ (from Device column in device_time.csv and monthly_device_capacity.csv)

Parameters:
- $p_i$ = unit profit of product $i$ (from unit_product_profits.csv)
- $a_{ki}$ = processing time required by product $i$ on device $k$ (from device_time.csv)
- $c_k$ = monthly operating capacity of device $k$ (from monthly_device_capacity.csv)

Decision variables:
- $x_i \geq 0$ = production quantity of product $i$ (continuous)

### Objective
$$
\max \sum_{i \in I} p_i x_i
$$

### Constraints

For each device $k \in K$:
$$
\sum_{i \in I} a_{ki} x_i \leq c_k
$$

For each product $i \in I$:
$$
x_i \geq 0
$$

### Data Mapping

- $I$ (products): file_2_view_0.Product, file_0_view_0 columns (excluding Device)
- $K$ (devices): file_0_view_0.Device, file_1_view_0.Device
- $p_i$: file_2_view_0.Unit_Profit, keyed by Product
- $a_{ki}$: file_0_view_0, value at row Device $k$, column Product $i$
- $c_k$: file_1_view_0.Monthly_Capacity, keyed by Device

### Summary

- Maximize total profit from all products.
- Each device's total processing time used by all products cannot exceed its monthly capacity.
- Production quantities are nonnegative and continuous.