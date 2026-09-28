#### Index Sets
- $\mathcal{P}$: set of products (from unit_product_profits.csv, device_time.csv), indexed by $p$
- $\mathcal{D}$: set of devices (from device_time.csv, monthly_device_capacity.csv), indexed by $d$

#### Parameters
- $u_p$: unit profit of product $p$ (from unit_product_profits.csv, column "Unit_Profit")
- $t_{d,p}$: processing time required on device $d$ per unit of product $p$ (from device_time.csv, columns "Device", "P1"..."P111")
- $c_d$: monthly capacity of device $d$ (from monthly_device_capacity.csv, column "Monthly_Capacity")

#### Decision Variables
- $x_p \geq 0$: continuous production quantity of product $p$ to produce in the month

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

- unit_product_profits.csv: 
    - Table ID: file_2_view_0
    - Columns: "Product" (index set $\mathcal{P}$), "Unit_Profit" (parameter $u_p$)
- device_time.csv: 
    - Table ID: file_0_view_0
    - Columns: "Device" (index set $\mathcal{D}$), "P1"..."P111" (parameter $t_{d,p}$)
- monthly_device_capacity.csv: 
    - Table ID: file_1_view_0
    - Columns: "Device" (index set $\mathcal{D}$), "Monthly_Capacity" (parameter $c_d$)