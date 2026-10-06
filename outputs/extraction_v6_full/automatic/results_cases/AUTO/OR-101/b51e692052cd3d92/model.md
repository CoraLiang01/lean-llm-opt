## Abstract Mathematical Model

Let:
- $I$ = set of products, indexed by $i$ (from Product column in unit_product_profits.csv and device_time.csv)
- $K$ = set of devices, indexed by $k$ (from Device column in device_time.csv and monthly_device_capacity.csv)
- $p_i$ = unit profit of product $i$ (from Unit_Profit in unit_product_profits.csv)
- $a_{ki}$ = processing time required by product $i$ on device $k$ (from device_time.csv, Device $k$, Product $i$)
- $c_k$ = monthly capacity of device $k$ (from Monthly_Capacity in monthly_device_capacity.csv)
- $x_i$ = production quantity of product $i$ (decision variable, continuous, $\geq 0$)

### Objective
$$
\max \sum_{i \in I} p_i x_i
$$

### Constraints

#### Device Capacity Constraints
$$
\sum_{i \in I} a_{ki} x_i \leq c_k \qquad \forall k \in K
$$

#### Nonnegativity
$$
x_i \geq 0 \qquad \forall i \in I
$$

---

## Data Mapping

- $I$ (products): file_2_view_0.Product, file_0_view_0.P* (columns P1–P111)
- $K$ (devices): file_0_view_0.Device, file_1_view_0.Device
- $p_i$: file_2_view_0.Unit_Profit, indexed by file_2_view_0.Product
- $a_{ki}$: file_0_view_0, value in column $i$ (P1–P111) for row Device $k$
- $c_k$: file_1_view_0.Monthly_Capacity, indexed by file_1_view_0.Device

- Decision variables $x_i$ are continuous and nonnegative, representing the monthly production quantity of product $i$.

---

## Summary

- Maximize total profit from all products.
- For each device, total processing time used by all products cannot exceed that device's monthly capacity.
- All production quantities are nonnegative and continuous.