#### Abstract Mathematical Model

**Index Sets:**
- $M$: set of radio models (from "HiFi1" to "HiFi101"), indexed by $m$
- $W$: set of workstations (from "1" to "3"), indexed by $w$

**Parameters:**
- $t_{w,m}$: processing time (in minutes) required for one unit of model $m$ at workstation $w$  
  (from `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture13/workstation_times.csv`, columns `HiFi*_Minutes`)
- $C$: total available minutes per workstation per day ($C = 1440$)
- $q_w$: maintenance percentage at workstation $w$  
  (from `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture13/workstation_times.csv`, column `Maintenance_Percent`)
- $E_w$: effective daily capacity at workstation $w$, $E_w = C \cdot (1 - q_w/100)$

**Decision Variables:**
- $x_m \in \mathbb{Z}_{\geq 0}$: number of units of radio model $m$ to produce per day

- $s_w \geq 0$: idle time (in minutes) at workstation $w$

**Objective:**
\[
\min \sum_{w \in W} s_w
\]

**Constraints:**
1. **Idle time definition for each workstation:**
   \[
   s_w = E_w - \sum_{m \in M} t_{w,m} x_m, \quad \forall w \in W
   \]
2. **Nonnegativity of idle time:**
   \[
   s_w \geq 0, \quad \forall w \in W
   \]
3. **Nonnegativity and integrality of production:**
   \[
   x_m \in \mathbb{Z}_{\geq 0}, \quad \forall m \in M
   \]

---

#### Data Mapping

- $M$ (radio models): `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture13/workstation_times.csv`, columns `HiFi1_Minutes` through `HiFi101_Minutes`
- $W$ (workstations): `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture13/workstation_times.csv`, column `Workstation`
- $t_{w,m}$: `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture13/workstation_times.csv`, row `Workstation` = $w$, column `HiFi{n}_Minutes` for $m = $ "HiFi$n$"
- $q_w$: `/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture13/workstation_times.csv`, row `Workstation` = $w$, column `Maintenance_Percent`
- $C$: fixed at 1440 (per query)
- $E_w$: computed as $C \cdot (1 - q_w/100)$ for each $w$ using $q_w$ above

**All parameters and index sets are mapped directly from the supplied CSV columns and rows.**