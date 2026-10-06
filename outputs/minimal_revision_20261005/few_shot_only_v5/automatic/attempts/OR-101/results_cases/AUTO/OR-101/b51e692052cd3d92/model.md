**Abstract Mathematical Model**

**Index Sets**
- $P$: Set of products (from `unit_product_profits.csv`, column `Product`)
- $D$: Set of devices (from `device_time.csv` and `monthly_device_capacity.csv`, column `Device`)

**Parameters**
- $c_{d}$: Monthly capacity of device $d$  
  (from `monthly_device_capacity.csv`, column `Monthly_Capacity`, keyed by `Device`)
- $a_{d,p}$: Processing time required by product $p$ on device $d$  
  (from `device_time.csv`, column $p$, keyed by `Device`)
- $u_p$: Unit profit of product $p$  
  (from `unit_product_profits.csv`, column `Unit_Profit`, keyed by `Product`)

**Decision Variables**
- $x_p \geq 0$: Production quantity of product $p$ (continuous, unrestricted above)

---

**Objective**
\[
\max \sum_{p \in P} u_p \, x_p
\]

**Constraints**
- **Device capacity constraints:**  
  For each device $d \in D$,
  \[
  \sum_{p \in P} a_{d,p} \, x_p \leq c_{d}
  \]
- **Nonnegativity:**  
  \[
  x_p \geq 0 \qquad \forall p \in P
  \]

---

**Data Mapping**

- $P$:  
  `unit_product_profits.csv`, column `Product`  
  `device_time.csv`, columns $P1$ through $P111$
- $D$:  
  `device_time.csv`, column `Device`  
  `monthly_device_capacity.csv`, column `Device`
- $u_p$:  
  `unit_product_profits.csv`, columns `Product`, `Unit_Profit`
- $a_{d,p}$:  
  `device_time.csv`, columns `Device`, $P1$ through $P111$
- $c_{d}$:  
  `monthly_device_capacity.csv`, columns `Device`, `Monthly_Capacity`

---

**Summary:**  
Maximize total profit from product mix, subject to device time capacities, with continuous nonnegative production variables for each product. All parameters and index sets are mapped directly to the provided CSV columns and business identifiers.