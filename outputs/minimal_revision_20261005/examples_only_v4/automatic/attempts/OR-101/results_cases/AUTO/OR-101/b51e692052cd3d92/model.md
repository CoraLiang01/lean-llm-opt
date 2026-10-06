**Abstract Mathematical Model**

**Index Sets:**
- $P$: Set of products (from `file_2_view_0.Product` and `file_0_view_0` columns)
- $D$: Set of devices (from `file_1_view_0.Device` and `file_0_view_0.Device`)

**Parameters:**
- $u_p$: Unit profit of product $p$  
  Data: `file_2_view_0.Unit_Profit`, keyed by `Product`
- $t_{dp}$: Processing time required by product $p$ on device $d$  
  Data: `file_0_view_0`, row `Device`, column $p$
- $c_d$: Monthly operating capacity of device $d$  
  Data: `file_1_view_0.Monthly_Capacity`, keyed by `Device$

**Decision Variables:**
- $x_p \geq 0$: Continuous quantity of product $p$ to produce

**Objective:**
\[
\max \sum_{p \in P} u_p \, x_p
\]

**Constraints:**
- Device capacity for each $d \in D$:
\[
\sum_{p \in P} t_{dp} \, x_p \leq c_d \qquad \forall d \in D
\]
- Nonnegativity:
\[
x_p \geq 0 \qquad \forall p \in P
\]

---

**Data Mapping**

- $P$: All products in `file_2_view_0.Product` and all product columns in `file_0_view_0` (P1–P111, matched by name)
- $D$: All devices in `file_1_view_0.Device` and `file_0_view_0.Device` (A–J, matched by name)
- $u_p$: `file_2_view_0.Unit_Profit`, keyed by `Product`
- $t_{dp}$: `file_0_view_0`, row `Device`, column $p$
- $c_d$: `file_1_view_0.Monthly_Capacity`, keyed by `Device`

**Variable Domain**
- $x_p \in \mathbb{R}_{\geq 0}$ (continuous, nonnegative)

**All index sets, parameters, and constraints are defined using the full set of records returned from CSVQA, preserving all original business identifiers and file mappings.**