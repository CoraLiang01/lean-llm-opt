**Abstract Mathematical Model**

**Index Sets**
- $M$: set of radio models (from HiFi-1 to HiFi-101; one for each column HiFi$k$_Minutes in workstation_times.csv)
- $W$: set of workstations (from the Workstation column in workstation_times.csv)

**Parameters**
- $t_{wm}$: processing time (in minutes) required for one unit of model $m \in M$ at workstation $w \in W$  
  (from column HiFi$k$_Minutes, row with Workstation $w$, in workstation_times.csv)
- $C_w$: total available minutes per day at workstation $w$ (given: $C_w = 1440$ for all $w$)
- $p_w$: maintenance percent at workstation $w$ (from Maintenance_Percent column, row with Workstation $w$)
- $E_w$: effective daily capacity at workstation $w$, $E_w = C_w \cdot (1 - p_w/100)$

**Decision Variables**
- $x_m \in \mathbb{Z}_{\geq 0}$: number of units of radio model $m \in M$ to produce per day

**Auxiliary Variables**
- $I_w \geq 0$: idle time (in minutes) at workstation $w \in W$

---

**Objective**
\[
\min \sum_{w \in W} I_w
\]

**Constraints**

1. **Idle time definition for each workstation:**
   \[
   I_w = E_w - \sum_{m \in M} t_{wm} x_m, \quad \forall w \in W
   \]
   (Idle time is effective capacity minus total processing time used.)

2. **Nonnegativity of idle time:**
   \[
   I_w \geq 0, \quad \forall w \in W
   \]

3. **Nonnegativity and integrality of production:**
   \[
   x_m \in \mathbb{Z}_{\geq 0}, \quad \forall m \in M
   \]

---

**Data Mapping**

- $M$: All columns in workstation_times.csv with names matching HiFi$k$_Minutes ($k=1,\ldots,101$)
- $W$: All values in the Workstation column of workstation_times.csv
- $t_{wm}$: Value in column HiFi$k$_Minutes, row with Workstation $w$, table_id: file_0_view_0
- $C_w$: Constant $1440$ for all $w$
- $p_w$: Value in Maintenance_Percent column, row with Workstation $w$, table_id: file_0_view_0
- $E_w$: $C_w \cdot (1 - p_w/100)$, computed per workstation
- $x_m$: Decision variable, indexed by $M$
- $I_w$: Auxiliary variable, indexed by $W$

**All parameters and index sets are defined using the exact columns and rows from workstation_times.csv (table_id: file_0_view_0).**