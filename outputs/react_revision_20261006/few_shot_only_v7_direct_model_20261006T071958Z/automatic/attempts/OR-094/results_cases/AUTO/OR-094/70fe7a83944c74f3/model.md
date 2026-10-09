ABSTRACT MATHEMATICAL MODEL

**Index Sets:**
- $M$: set of radio models (from all "HiFi$n$_Minutes" columns in file_0_view_0, $n=1,\ldots,101$)
- $W$: set of workstations (from "Workstation" column in file_0_view_0, $W = \{1,2,3\}$)

**Parameters:**
- $t_{w,m}$: processing time (in minutes) required at workstation $w$ for one unit of model $m$  
  (from file_0_view_0, column "HiFi$n$_Minutes" for model $m$ and row with "Workstation" = $w$)
- $C_w$: total available minutes per day at workstation $w$ (given as 1,440 for all $w$)
- $r_w$: maintenance percent at workstation $w$ (from file_0_view_0, column "Maintenance_Percent" for row with "Workstation" = $w$)
- $E_w$: effective daily capacity at workstation $w$, $E_w = C_w \cdot (1 - r_w/100)$

**Decision Variables:**
- $x_m \in \mathbb{Z}_{\geq 0}$: number of units of model $m$ to produce per day

**Auxiliary Variables:**
- $I_w \geq 0$: idle time (in minutes) at workstation $w$

**Objective:**
\[
\min \sum_{w \in W} I_w
\]

**Constraints:**
1. **Idle time definition for each workstation:**
   \[
   I_w = E_w - \sum_{m \in M} t_{w,m} x_m, \quad \forall w \in W
   \]
2. **Nonnegativity of idle time:**
   \[
   I_w \geq 0, \quad \forall w \in W
   \]
3. **Nonnegativity and integrality of production:**
   \[
   x_m \in \mathbb{Z}_{\geq 0}, \quad \forall m \in M
   \]

---

**DATA MAPPING**

- $M$: All columns in file_0_view_0 with names matching "HiFi$n$_Minutes", $n=1,\ldots,101$
- $W$: All values in file_0_view_0["Workstation"]
- $t_{w,m}$: file_0_view_0, row with "Workstation" = $w$, column $m$ ("HiFi$n$_Minutes")
- $C_w$: 1,440 for all $w$
- $r_w$: file_0_view_0, row with "Workstation" = $w$, column "Maintenance_Percent"
- $E_w$: $C_w \cdot (1 - r_w/100)$
- $x_m$: decision variable for model $m$
- $I_w$: auxiliary variable for workstation $w$

All index sets, parameters, and mappings are defined using the exact table_id and column names from the current Observation.