Let $x_k$ be the number of units of radio model $k$ (for $k = 1, \ldots, 101$) to produce per day. Let $t_{wk}$ be the processing time (in minutes) required for one unit of model $k$ at workstation $w$ (for $w = 1,2,3$). Let $C_w$ be the effective daily capacity (in minutes) of workstation $w$ after maintenance.

Define:
- $C_1 = 1440 \times (1 - 0.10) = 1296$
- $C_2 = 1440 \times (1 - 0.14) = 1238.4$
- $C_3 = 1440 \times (1 - 0.12) = 1267.2$

Minimize total idle time across all workstations:
$$
\min \sum_{w=1}^3 \left[ C_w - \sum_{k=1}^{101} t_{wk} x_k \right]
$$

Subject to:
\[
\sum_{k=1}^{101} t_{wk} x_k \leq C_w \qquad \forall w = 1,2,3
\]
\[
x_k \in \mathbb{Z}_{\geq 0} \qquad \forall k = 1, \ldots, 101
\]

Where the $t_{wk}$ coefficients are given by the following table (source order preserved):

| Workstation | HiFi1_Minutes | HiFi2_Minutes | ... | HiFi101_Minutes | Maintenance_Percent |
|-------------|--------------|--------------|-----|-----------------|--------------------|
| 1           | 6            | 4            | ... | 9               | 10                 |
| 2           | 5            | 5            | ... | 5               | 14                 |
| 3           | 4            | 6            | ... | 6               | 12                 |

(Use all 101 HiFi*_Minutes columns for $k=1$ to $101$; the full table is as retrieved.)

All variables $x_k$ are nonnegative integers. The objective is to minimize the sum of idle times at all three workstations, where idle time at workstation $w$ is $C_w - \sum_{k=1}^{101} t_{wk} x_k$.