ABSTRACT MATHEMATICAL MODEL

Index Sets:
- \( M \): Set of radio models (e.g., HiFi-1, ..., HiFi-101), indexed by \( m \).
- \( W \): Set of workstations (from the Workstation column), indexed by \( w \).

Parameters:
- \( p_{w,m} \): Processing time (in minutes) required at workstation \( w \) per unit of model \( m \).
- \( C_w \): Total available time per day at workstation \( w \) (1,440 minutes for all \( w \)).
- \( \alpha_w \): Maintenance percentage at workstation \( w \) (from Maintenance_Percent column).
- \( E_w = C_w \times (1 - \alpha_w/100) \): Effective daily production capacity (in minutes) at workstation \( w \).

Variables:
- \( x_m \in \mathbb{Z}_+ \): Number of units of model \( m \) to produce per day (nonnegative integer).
- \( \text{Idle}_w \geq 0 \): Idle production time (in minutes) at workstation \( w \).

Objective:
\[
\min \sum_{w \in W} \text{Idle}_w
\]
where for each \( w \):
\[
\text{Idle}_w = E_w - \sum_{m \in M} p_{w,m} x_m
\]

Constraints:
1. Idle time definition and nonnegativity:
   \[
   \text{Idle}_w = E_w - \sum_{m \in M} p_{w,m} x_m \quad \forall w \in W
   \]
   \[
   \text{Idle}_w \geq 0 \quad \forall w \in W
   \]
2. Production cannot exceed effective capacity:
   \[
   \sum_{m \in M} p_{w,m} x_m \leq E_w \quad \forall w \in W
   \]
3. Nonnegativity and integrality:
   \[
   x_m \in \mathbb{Z}_+, \quad \forall m \in M
   \]

Data Mapping:

- \( W \): All values in file_0_view_0.Workstation (original row order).
- \( M \): All model columns in file_0_view_0 with names matching pattern "HiFi*_Minutes".
- \( p_{w,m} \): file_0_view_0.[HiFi*_Minutes] for each workstation \( w \) and model \( m \).
- \( C_w \): 1,440 for all \( w \).
- \( \alpha_w \): file_0_view_0.Maintenance_Percent for each workstation \( w \).
- \( E_w \): \( 1,440 \times (1 - \)file_0_view_0.Maintenance_Percent\(/100) \) for each workstation \( w \).

- Decision variables \( x_m \) are indexed by model columns (e.g., HiFi1_Minutes, ..., HiFi101_Minutes).
- Each constraint and parameter is mapped using the explicit Workstation and model column names from file_0_view_0.

This model minimizes the total idle production time across all workstations, subject to effective capacity limits and integer production decisions for each radio model.