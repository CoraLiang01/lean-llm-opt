#### Abstract Mathematical Model

Let:
- $I$ = set of products, indexed by $i$ (from Product column in unit_product_profits.csv and device_time.csv)
- $K$ = set of devices, indexed by $k$ (from Device column in device_time.csv and monthly_device_capacity.csv)
- $p_i$ = unit profit of product $i$ (from Unit_Profit in unit_product_profits.csv)
- $a_{ki}$ = processing time required by product $i$ on device $k$ (from device_time.csv)
- $c_k$ = monthly capacity of device $k$ (from Monthly_Capacity in monthly_device_capacity.csv)
- $x_i$ = production quantity of product $i$ (decision variable, continuous, $x_i \geq 0$)

**Objective:**
$$
\max \sum_{i \in I} p_i x_i
$$

**Subject to:**
- Device capacity constraints:
$$
\sum_{i \in I} a_{ki} x_i \leq c_k, \quad \forall k \in K
$$

- Nonnegativity:
$$
x_i \geq 0, \quad \forall i \in I
$$

---

#### Data Mapping

- $I$ (Products): file_2_view_0.Product, file_0_view_0.[P1,...,P111]
- $K$ (Devices): file_0_view_0.Device, file_1_view_0.Device
- $p_i$: file_2_view_0.Unit_Profit, keyed by file_2_view_0.Product
- $a_{ki}$: file_0_view_0.[P1,...,P111], keyed by file_0_view_0.Device (rows) and product columns
- $c_k$: file_1_view_0.Monthly_Capacity, keyed by file_1_view_0.Device

- Decision variables $x_i$: continuous, nonnegative, for each $i \in I$ (product from Product column in unit_product_profits.csv)

**All parameters and indices are mapped directly from the original files and columns as described above.**