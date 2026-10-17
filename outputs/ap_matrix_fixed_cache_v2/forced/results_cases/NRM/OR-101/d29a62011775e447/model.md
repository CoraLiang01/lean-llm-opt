#### Index Sets
- $\mathcal{P}$: set of products (from Product in unit_product_profits.csv and device_time.csv), e.g., $\mathcal{P} = \{\text{P1}, \ldots, \text{P111}\}$
- $\mathcal{D}$: set of devices (from Device in device_time.csv and monthly_device_capacity.csv), e.g., $\mathcal{D} = \{\text{A}, \ldots, \text{J}\}$

#### Parameters
- $u_p$: unit profit of product $p \in \mathcal{P}$  
  (from file_2_view_0, column Unit_Profit, key Product)
- $t_{d,p}$: processing time required on device $d \in \mathcal{D}$ per unit of product $p \in \mathcal{P}$  
  (from file_0_view_0, value at [Device $d$, column $p$])
- $c_d$: monthly capacity of device $d \in \mathcal{D}$  
  (from file_1_view_0, column Monthly_Capacity, key Device)

#### Decision Variables
- $x_p \geq 0$: continuous production quantity of product $p \in \mathcal{P}$

#### Objective
$$
\max \sum_{p \in \mathcal{P}} u_p \, x_p
$$

#### Constraints
- Device capacity constraints (for all $d \in \mathcal{D}$):
$$
\sum_{p \in \mathcal{P}} t_{d,p} \, x_p \leq c_d
$$
- Nonnegativity (for all $p \in \mathcal{P}$):
$$
x_p \geq 0
$$

---

#### Data Mapping

- $\mathcal{P}$: All Product values in file_2_view_0 (unit_product_profits.csv, column Product) and file_0_view_0 (device_time.csv, columns P1–P111)
- $\mathcal{D}$: All Device values in file_0_view_0 (device_time.csv, column Device) and file_1_view_0 (monthly_device_capacity.csv, column Device)
- $u_p$: file_2_view_0 (unit_product_profits.csv), column Unit_Profit, key Product
- $t_{d,p}$: file_0_view_0 (device_time.csv), value at [Device $d$, column $p$]
- $c_d$: file_1_view_0 (monthly_device_capacity.csv), column Monthly_Capacity, key Device

All rows and columns from the three files are used as returned by CSVQA.