**Abstract Mathematical Model**

**Index Sets**
- $P$: Set of products (from `file_2_view_0`, column `Product`)
- $D$: Set of devices (from `file_1_view_0`, column `Device`)

**Parameters**
- $u_p$: Unit profit of product $p$  
  Data: `file_2_view_0`, columns `Product`, `Unit_Profit`
- $t_{d,p}$: Processing time required by product $p$ on device $d$  
  Data: `file_0_view_0`, row `Device`, column $p$
- $c_d$: Monthly operating capacity of device $d$  
  Data: `file_1_view_0`, columns `Device`, `Monthly_Capacity$

**Decision Variables**
- $x_p \geq 0$: Production quantity of product $p$ (continuous)

**Objective**
\[
\max \sum_{p \in P} u_p \, x_p
\]

**Constraints**
- Device capacity constraints (for all $d \in D$):
\[
\sum_{p \in P} t_{d,p} \, x_p \leq c_d
\]
- Nonnegativity:
\[
x_p \geq 0 \qquad \forall p \in P
\]

---

**Data Mapping**

- $P$: All values in `file_2_view_0`, column `Product`
- $D$: All values in `file_1_view_0`, column `Device`
- $u_p$: `file_2_view_0`, columns `Product`, `Unit_Profit`
- $t_{d,p}$: `file_0_view_0`, row `Device`, column $p$ (for all $d \in D$, $p \in P$)
- $c_d$: `file_1_view_0`, columns `Device`, `Monthly_Capacity` (for all $d \in D$)
- $x_p$: Decision variable for each $p \in P$ (continuous, nonnegative)