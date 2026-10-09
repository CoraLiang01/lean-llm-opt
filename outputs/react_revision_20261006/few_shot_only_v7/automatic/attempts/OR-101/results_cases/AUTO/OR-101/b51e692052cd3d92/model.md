#### Mathematical Model

Let:
- $I$ = set of products (indexed by $i$), from column Product in unit_product_profits.csv and device_time.csv
- $K$ = set of devices (indexed by $k$), from column Device in device_time.csv and monthly_device_capacity.csv
- $x_i \geq 0$ = production quantity of product $i$ (continuous variable)

Parameters:
- $p_i$ = unit profit of product $i$ (from unit_product_profits.csv, column Unit_Profit)
- $a_{ki}$ = processing time required by product $i$ on device $k$ (from device_time.csv, columns P1–P111)
- $c_k$ = monthly capacity of device $k$ (from monthly_device_capacity.csv, column Monthly_Capacity)

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

- $I$ (products): unit_product_profits.csv, column Product; device_time.csv, columns P1–P111
- $K$ (devices): device_time.csv, column Device; monthly_device_capacity.csv, column Device
- $p_i$: unit_product_profits.csv, column Unit_Profit, keyed by Product
- $a_{ki}$: device_time.csv, value at row with Device $k$, column $i$ (P1–P111)
- $c_k$: monthly_device_capacity.csv, column Monthly_Capacity, keyed by Device
- $x_i$: production quantity of product $i$ (decision variable, continuous, nonnegative)