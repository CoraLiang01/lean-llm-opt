#### Abstract Mathematical Model

**Index Sets:**
- $M$: set of radio models (from HiFi-1 to HiFi-101; see Data Mapping)
- $W$: set of workstations (from workstation_times.csv; see Data Mapping)

**Parameters:**
- $t_{w,m}$: processing time (in minutes) required per unit of model $m$ at workstation $w$ (from workstation_times.csv)
- $C_w$: total available time per day at workstation $w$ (given: $C_w = 1440$ for all $w$)
- $p_w$: maintenance percentage at workstation $w$ (from workstation_times.csv)
- $E_w$: effective daily capacity at workstation $w$, $E_w = C_w \cdot (1 - p_w/100)$

**Decision Variables:**
- $x_m \in \mathbb{Z}_{\geq 0}$: number of units of model $m$ to produce per day

**Auxiliary Variables:**
- $I_w \geq 0$: idle time (in minutes) at workstation $w$

---

**Objective:**
$$
\min \sum_{w \in W} I_w
$$

**Constraints:**
1. **Idle time definition for each workstation:**
   $$
   I_w = E_w - \sum_{m \in M} t_{w,m} x_m, \quad \forall w \in W
   $$
2. **Nonnegativity of idle time:**
   $$
   I_w \geq 0, \quad \forall w \in W
   $$
3. **Nonnegativity and integrality of production:**
   $$
   x_m \in \mathbb{Z}_{\geq 0}, \quad \forall m \in M
   $$

---

#### Data Mapping

- **Radio Models ($M$):** Each column in workstation_times.csv with name HiFiX_Minutes, where $X$ is 1 to 101, corresponds to model $m$.
- **Workstations ($W$):** Each row in workstation_times.csv, identified by column Workstation.
- **Processing Times ($t_{w,m}$):** Value in column HiFiX_Minutes for row with Workstation $w$ and model $m$.
- **Maintenance Percent ($p_w$):** Value in column Maintenance_Percent for row with Workstation $w$.
- **Effective Capacity ($E_w$):** $E_w = 1440 \times (1 - p_w/100)$ for each $w$.

- **Source Table:** workstation_times.csv
  - Workstation: $w$
  - HiFiX_Minutes: $t_{w,m}$ for model $m$ = HiFi-X
  - Maintenance_Percent: $p_w$

---

**All parameters and index sets are mapped directly from workstation_times.csv as described above.**