ABSTRACT MATHEMATICAL MODEL

Index Sets:
- 𝑊: Set of workstations (indexed by 𝑤), from workstation_times.csv, column Workstation.
- 𝑀: Set of radio models (indexed by 𝑚), where each 𝑚 corresponds to a column HiFiX_Minutes (X = 1,…,101).

Parameters:
- 𝑝_{w,m}: Processing time (in minutes) required at workstation 𝑤 per unit of model 𝑚.  
- 𝑀𝑎𝑖𝑛𝑡_{w}: Maintenance percent at workstation 𝑤 (as a percentage, e.g., 10 for 10%), from Maintenance_Percent.
- 𝐶𝑎𝑝: Total available time per workstation per day (1,440 minutes).
- 𝐸𝑓𝑓𝐶𝑎𝑝_{w}: Effective daily production capacity at workstation 𝑤, computed as 𝐶𝑎𝑝 × (1 - 𝑀𝑎𝑖𝑛𝑡_{w}/100).

Decision Variables:
- 𝑥_{m} ∈ ℤ₊: Number of units of model 𝑚 to produce per day (nonnegative integer).
- 𝑖_{w} ≥ 0: Idle production time (in minutes) at workstation 𝑤 (continuous, nonnegative).

Objective:
Minimize total idle production time across all workstations:
\[
\min \sum_{w \in W} i_{w}
\]

Constraints:
1. Idle time definition for each workstation:
   \[
   i_{w} = \text{EffCap}_{w} - \sum_{m \in M} p_{w,m} \, x_{m} \quad \forall w \in W
   \]
2. Nonnegativity of idle time:
   \[
   i_{w} \geq 0 \quad \forall w \in W
   \]
3. Production cannot exceed effective capacity:
   \[
   \sum_{m \in M} p_{w,m} \, x_{m} \leq \text{EffCap}_{w} \quad \forall w \in W
   \]
4. Nonnegativity and integrality of production:
   \[
   x_{m} \in \mathbb{Z}_{+} \quad \forall m \in M
   \]

DATA MAPPING

Index Sets:
- 𝑊: All rows in file_0_view_0, column Workstation.
- 𝑀: All columns in file_0_view_0 with names matching HiFiX_Minutes (X = 1,…,101).

Parameters:
- 𝑝_{w,m}: file_0_view_0, row with Workstation = 𝑤, column = 𝑚.
- 𝑀𝑎𝑖𝑛𝑡_{w}: file_0_view_0, row with Workstation = 𝑤, column Maintenance_Percent.
- 𝐶𝑎𝑝: 1,440 (constant, per query).
- 𝐸𝑓𝑓𝐶𝑎𝑝_{w}: 1,440 × (1 - Maintenance_Percent/100), file_0_view_0, row with Workstation = 𝑤.

Decision Variables:
- 𝑥_{m}: For each 𝑚 ∈ 𝑀 (HiFiX_Minutes columns).
- 𝑖_{w}: For each 𝑤 ∈ 𝑊 (Workstation).

All data is sourced from file_0_view_0 (workstation_times.csv) as described above.