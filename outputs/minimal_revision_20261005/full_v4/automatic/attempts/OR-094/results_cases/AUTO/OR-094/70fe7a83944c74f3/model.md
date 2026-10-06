**Abstract Mathematical Model**

**Index Sets**
- $W$: set of workstations, indexed by $w$ (from all Workstation values in file_0_view_0)
- $M$: set of radio models, indexed by $m$ (from all HiFi* columns in file_0_view_0)

**Parameters**
- $T$: total available minutes per workstation per day (given: $T = 1440$)
- $q_w$: maintenance percentage at workstation $w$ (from Maintenance_Percent in file_0_view_0)
- $c_w$: effective daily capacity at workstation $w$, $c_w = T \cdot (1 - q_w/100)$
- $a_{w,m}$: processing time (in minutes) required at workstation $w$ per unit of model $m$ (from column $m$ in row with Workstation $w$ in file_0_view_0)

**Decision Variables**
- $x_m \in \mathbb{Z}_{\geq 0}$: number of units of model $m$ to produce per day

**Auxiliary Variables**
- $y_w \geq 0$: idle time (in minutes) at workstation $w$

---

**Objective**
\[
\min \sum_{w \in W} y_w
\]

**Constraints**
1. **Idle time definition for each workstation:**
   \[
   y_w = c_w - \sum_{m \in M} a_{w,m} x_m \qquad \forall w \in W
   \]
2. **Nonnegativity of idle time:**
   \[
   y_w \geq 0 \qquad \forall w \in W
   \]
3. **Nonnegativity and integrality of production:**
   \[
   x_m \in \mathbb{Z}_{\geq 0} \qquad \forall m \in M
   \]

---

**Data Mapping**

- $W$: All Workstation values from file_0_view_0, column Workstation
- $M$: All radio model columns from file_0_view_0, columns matching pattern HiFi*Minutes
- $q_w$: file_0_view_0, column Maintenance_Percent, for each $w$
- $a_{w,m}$: file_0_view_0, value in column $m$ for row with Workstation $w$
- $T$: fixed at 1440
- $c_w$: $T \cdot (1 - q_w/100)$ for each $w$
- $x_m$: decision variable for each $m$
- $y_w$: auxiliary variable for each $w$

**Source Table(s):**
- file_0_view_0: columns Workstation, HiFi1_Minutes, ..., HiFi101_Minutes, Maintenance_Percent

---

**Summary:**  
Minimize total idle time across all workstations by choosing integer daily production quantities for each radio model, subject to workstation effective capacities and per-model processing times. All parameters and index sets are mapped directly from the supplied data.