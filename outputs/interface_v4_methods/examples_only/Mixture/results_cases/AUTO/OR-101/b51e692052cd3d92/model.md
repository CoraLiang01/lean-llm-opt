## Abstract Mathematical Model

**Index Sets:**
- $P$: set of products (from `unit_product_profits.csv` and `device_time.csv`, column "Product", e.g., $P = \{\text{P1}, \ldots, \text{P111}\}$)
- $D$: set of devices (from `device_time.csv` and `monthly_device_capacity.csv`, column "Device", e.g., $D = \{\text{A}, \ldots, \text{J}\}$)

**Parameters:**
- $u_p$: unit profit of product $p \in P$  
  (from `unit_product_profits.csv`, column "Unit_Profit", key "Product")
- $t_{dp}$: processing time required by product $p$ on device $d$  
  (from `device_time.csv`, value at row with "Device" = $d$, column $p$)
- $c_d$: monthly capacity of device $d$  
  (from `monthly_device_capacity.csv`, column "Monthly_Capacity", key "Device")

**Decision Variables:**
- $x_p \geq 0$: continuous quantity of product $p$ to produce in the month

**Objective:**
\[
\max \sum_{p \in P} u_p \, x_p
\]

**Constraints:**
\[
\sum_{p \in P} t_{dp} \, x_p \leq c_d \qquad \forall d \in D
\]
\[
x_p \geq 0 \qquad \forall p \in P
\]

---

## Data Mapping

- $u_p$:  
  - Table: `unit_product_profits.csv` (`file_2_view_0`)  
  - Key: "Product"  
  - Value: "Unit_Profit"

- $t_{dp}$:  
  - Table: `device_time.csv` (`file_0_view_0`)  
  - Row key: "Device"  
  - Column key: product columns "P1"–"P111"  
  - Value: cell at ("Device" = $d$, product column = $p$)

- $c_d$:  
  - Table: `monthly_device_capacity.csv` (`file_1_view_0`)  
  - Key: "Device"  
  - Value: "Monthly_Capacity"

---

**All index sets, parameters, and constraints are mapped directly from the original files and columns as described above.**