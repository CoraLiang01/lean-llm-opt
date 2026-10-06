#### Abstract Mathematical Model

**Index Sets:**
- $M$: set of radio models (from HiFi-1 to HiFi-101), indexed by $m$
- $W$: set of workstations (from file_0_view_0.Workstation), indexed by $w$

**Parameters:**
- $t_{w,m}$: processing time (in minutes) required per unit of model $m$ at workstation $w$  
  (from file_0_view_0, column for $m$ in row with Workstation $w$)
- $C_w$: total available minutes per day at workstation $w$ (given: $1,440$ for all $w$)
- $q_w$: maintenance percent at workstation $w$ (from file_0_view_0.Maintenance_Percent, row with Workstation $w$)
- $E_w$: effective daily capacity at workstation $w$, $E_w = C_w \cdot (1 - q_w/100)$

**Decision Variables:**
- $x_m \in \mathbb{Z}_{\geq 0}$: number of units of model $m$ to produce per day

**Auxiliary Variables:**
- $I_w \geq 0$: idle time (in minutes) at workstation $w$

**Objective:**
$$
\min \sum_{w \in W} I_w
$$

**Constraints:**
1. Idle time definition for each workstation:
   $$
   I_w = E_w - \sum_{m \in M} t_{w,m} x_m, \quad \forall w \in W
   $$
2. Nonnegativity of idle time:
   $$
   I_w \geq 0, \quad \forall w \in W
   $$
3. Nonnegativity and integrality of production:
   $$
   x_m \in \mathbb{Z}_{\geq 0}, \quad \forall m \in M
   $$

---

#### Data Mapping

- $M$:  
  All model columns in file_0_view_0 with names matching pattern "HiFi[1-101]_Minutes"
- $W$:  
  file_0_view_0.Workstation
- $t_{w,m}$:  
  file_0_view_0, row with Workstation $w$, column $m$ (e.g., "HiFi1_Minutes", ..., "HiFi101_Minutes")
- $q_w$:  
  file_0_view_0, row with Workstation $w$, column "Maintenance_Percent"
- $C_w$:  
  Constant $1,440$ for all $w$
- $E_w$:  
  $E_w = 1,440 \times (1 - \text{file\_0\_view\_0.Maintenance\_Percent}/100)$ for each $w$
- $x_m$:  
  Decision variable for each $m$ in $M$
- $I_w$:  
  Auxiliary variable for each $w$ in $W$

---

**All parameters and index sets are mapped directly from file_0_view_0 (workstation_times.csv) using the original column and row identifiers.**