### Mathematical Model

Let:
- $I$ = set of products, indexed by $i$ (from all Product in unit_product_profits.csv and device_time.csv)
- $K$ = set of devices, indexed by $k$ (from all Device in monthly_device_capacity.csv and device_time.csv)
- $x_i \geq 0$ = production quantity of product $i$ (continuous)

Parameters:
- $p_i$ = unit profit of product $i$ (from unit_product_profits.csv)
- $a_{ki}$ = processing time required on device $k$ per unit of product $i$ (from device_time.csv)
- $c_k$ = monthly capacity of device $k$ (from monthly_device_capacity.csv)

Objective:
\[
\max \sum_{i \in I} p_i x_i
\]

Subject to (for all $k \in K$):
\[
\sum_{i \in I} a_{ki} x_i \leq c_k
\]
\[
x_i \geq 0 \quad \forall i \in I
\]

---

### Data Mapping

- $I$ (products): All Product in file_2_view_0 (unit_product_profits.csv) and columns of file_0_view_0 (device_time.csv) except "Device"
- $K$ (devices): All Device in file_1_view_0 (monthly_device_capacity.csv) and file_0_view_0 (device_time.csv)
- $p_i$: file_2_view_0, column Unit_Profit, key Product
- $a_{ki}$: file_0_view_0, value at row Device $k$, column $i$
- $c_k$: file_1_view_0, column Monthly_Capacity, key Device

- Decision variables: $x_i$ (continuous, nonnegative), for all $i \in I$.

All indices, parameters, and constraints are mapped directly to the original CSV columns and business identifiers.