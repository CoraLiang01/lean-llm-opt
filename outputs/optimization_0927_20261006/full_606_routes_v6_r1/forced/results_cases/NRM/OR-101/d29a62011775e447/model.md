#### Abstract Mathematical Model

**Index Sets:**
- $\mathcal{P}$: set of products (from P1 to P111)
- $\mathcal{D}$: set of devices (from A to J)

**Parameters:**
- $u_p$: unit profit of product $p \in \mathcal{P}$  
  [from file_2_view_0, column "Unit_Profit"]
- $t_{d,p}$: processing time required on device $d \in \mathcal{D}$ per unit of product $p \in \mathcal{P}$  
  [from file_0_view_0, columns "Device", "P1"..."P111"]
- $c_d$: monthly operating capacity of device $d \in \mathcal{D}$  
  [from file_1_view_0, column "Monthly_Capacity"]

**Decision Variables:**
- $x_p \geq 0$: continuous production quantity of product $p \in \mathcal{P}$

**Objective:**
\[
\max \sum_{p \in \mathcal{P}} u_p \, x_p
\]

**Constraints:**
- Device capacity constraints (for all $d \in \mathcal{D}$):
\[
\sum_{p \in \mathcal{P}} t_{d,p} \, x_p \leq c_d
\]
- Nonnegativity (for all $p \in \mathcal{P}$):
\[
x_p \geq 0
\]

---

#### Data Mapping

- $\mathcal{P}$: All "Product" values in file_2_view_0 (unit_product_profits.csv, column "Product")
- $\mathcal{D}$: All "Device" values in file_1_view_0 (monthly_device_capacity.csv, column "Device")
- $u_p$: file_2_view_0 (unit_product_profits.csv), column "Unit_Profit", keyed by "Product"
- $t_{d,p}$: file_0_view_0 (device_time.csv), columns "Device" (rows) and "P1"..."P111" (columns)
- $c_d$: file_1_view_0 (monthly_device_capacity.csv), column "Monthly_Capacity", keyed by "Device"

All data is used as returned by CSVQA, with no additional filtering or transformation.