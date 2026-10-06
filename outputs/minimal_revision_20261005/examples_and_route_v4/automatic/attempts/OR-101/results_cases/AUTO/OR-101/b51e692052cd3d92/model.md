**Abstract Mathematical Model**

**Index Sets**
- $P$: Set of products (from `file_2_view_0`, column `Product`)
- $D$: Set of devices (from `file_1_view_0`, column `Device`)

**Parameters**
- $c_p$: Unit profit of product $p$  
  &nbsp;&nbsp;&nbsp;&nbsp;Data: `file_2_view_0`, columns `Product`, `Unit_Profit`
- $a_{dp}$: Processing time required by product $p$ on device $d$  
  &nbsp;&nbsp;&nbsp;&nbsp;Data: `file_0_view_0`, row `Device`, column $p$
- $b_d$: Monthly operating capacity of device $d$  
  &nbsp;&nbsp;&nbsp;&nbsp;Data: `file_1_view_0`, columns `Device`, `Monthly_Capacity`

**Decision Variables**
- $x_p \geq 0$: Continuous, quantity of product $p$ to produce in the month

**Objective**
\[
\max \sum_{p \in P} c_p \, x_p
\]

**Constraints**
- **Device capacity constraints:**  
  For each device $d \in D$,
  \[
  \sum_{p \in P} a_{dp} \, x_p \leq b_d
  \]
- **Nonnegativity:**  
  \[
  x_p \geq 0 \qquad \forall p \in P
  \]

---

**Data Mapping**

- $P$: All values in `file_2_view_0`, column `Product`
- $D$: All values in `file_1_view_0`, column `Device`
- $c_p$: `file_2_view_0`, columns `Product`, `Unit_Profit`
- $a_{dp}$: `file_0_view_0`, row `Device`, column $p$ (for all $d \in D$, $p \in P$)
- $b_d$: `file_1_view_0`, columns `Device`, `Monthly_Capacity`