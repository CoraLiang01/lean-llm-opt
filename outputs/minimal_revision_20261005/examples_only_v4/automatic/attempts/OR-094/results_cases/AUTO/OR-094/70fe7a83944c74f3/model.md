**Abstract Mathematical Model**

**Index Sets**
- $M$: Set of radio models (from "HiFi1" to "HiFi101"), indexed by $m$.
- $W$: Set of workstations (from "1" to "3"), indexed by $w$.

**Parameters**
- $t_{w,m}$: Processing time (in minutes) required at workstation $w$ for one unit of model $m$.
- $C_w$: Total available minutes per day at workstation $w$ (from query: $C_w = 1440$ for all $w$).
- $q_w$: Fraction of time consumed by maintenance at workstation $w$ (from "Maintenance_Percent" in data; $q_w = \text{Maintenance_Percent}/100$).
- $E_w$: Effective daily capacity at workstation $w$, $E_w = C_w \cdot (1 - q_w)$.

**Decision Variables**
- $x_m \in \mathbb{Z}_{\geq 0}$: Number of units of radio model $m$ to produce per day.

**Auxiliary Variables**
- $I_w \geq 0$: Idle time (in minutes) at workstation $w$ per day.

---

**Objective**
\[
\min \sum_{w \in W} I_w
\]

**Constraints**

1. **Idle Time Definition (per workstation):**
   \[
   I_w = E_w - \sum_{m \in M} t_{w,m} x_m, \quad \forall w \in W
   \]
   (Alternatively, $I_w \geq E_w - \sum_{m \in M} t_{w,m} x_m$, but since $x_m$ are integer and $I_w$ is continuous and can be zero, equality is valid.)

2. **Non-negativity of Idle Time:**
   \[
   I_w \geq 0, \quad \forall w \in W
   \]

3. **Non-negativity and Integrality of Production:**
   \[
   x_m \in \mathbb{Z}_{\geq 0}, \quad \forall m \in M
   \]

4. **No workstation may be overloaded:**
   \[
   \sum_{m \in M} t_{w,m} x_m \leq E_w, \quad \forall w \in W
   \]
   (This is implied by $I_w \geq 0$ and the definition above, but can be included for clarity.)

---

**Data Mapping**

- $t_{w,m}$: From `workstation_times.csv`, column `"Workstation"` = $w$, column `"HiFiX_Minutes"` = $t_{w,m}$ for $m$ corresponding to "HiFiX".
- $q_w$: From `workstation_times.csv`, column `"Workstation"` = $w$, column `"Maintenance_Percent"` divided by 100.
- $C_w$: Query-defined, $C_w = 1440$ for all $w$.
- $E_w$: $E_w = C_w \cdot (1 - q_w)$, computed per workstation.

**Index Set Mapping**

- $W$: All values of `"Workstation"` in `workstation_times.csv`.
- $M$: All radio models corresponding to columns `"HiFi1_Minutes"` through `"HiFi101_Minutes"` in `workstation_times.csv`.

**Variable Mapping**

- $x_m$: Number of units of model $m$ to produce per day.
- $I_w$: Idle time at workstation $w$ per day.

---

**Summary**

- **Minimize** total idle time across all workstations.
- **Decide** integer daily production quantities for each radio model.
- **Subject to**: For each workstation, total processing time used by all models does not exceed its effective daily capacity (after maintenance), and idle time is the unused portion of that capacity. All variables are nonnegative; production is in whole units.

**Data Mapping Table**

| Symbol         | Source Table (table_id)      | Column(s) Used                | Indexing Key(s)         |
|----------------|-----------------------------|-------------------------------|-------------------------|
| $t_{w,m}$      | file_0_view_0               | "Workstation", "HiFiX_Minutes"| $w$ = "Workstation", $m$ = "HiFiX" |
| $q_w$          | file_0_view_0               | "Workstation", "Maintenance_Percent" | $w$ = "Workstation"    |
| $C_w$          | (query)                     | (fixed at 1440)               | $w$                    |
| $E_w$          | (computed)                  | $C_w \cdot (1 - q_w)$         | $w$                    |
| $x_m$          | (decision variable)         |                               | $m$                    |
| $I_w$          | (auxiliary variable)        |                               | $w$                    |

**All index sets, parameters, and constraints are mapped directly to the supplied data.**