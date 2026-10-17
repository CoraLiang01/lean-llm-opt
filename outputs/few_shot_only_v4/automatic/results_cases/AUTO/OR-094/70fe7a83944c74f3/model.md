Let $x_k$ be the number of units of radio model $k$ (for $k = 1, \ldots, 101$) to produce per day. Let $I_w$ be the idle time (in minutes) at workstation $w$ ($w = 1,2,3$).

Define:
- $t_{w,k}$ = processing time (in minutes) required at workstation $w$ per unit of model $k$ (from the table below).
- Each workstation has a total of 1,440 minutes per day.
- Effective available time at each workstation:
  - Workstation 1: $C_1 = 1,440 \times (1 - 0.10) = 1,296$ minutes
  - Workstation 2: $C_2 = 1,440 \times (1 - 0.14) = 1,238.4$ minutes
  - Workstation 3: $C_3 = 1,440 \times (1 - 0.12) = 1,267.2$ minutes

#### Decision Variables

- $x_k \in \mathbb{Z}_{\geq 0}$, for $k = 1, \ldots, 101$ (number of units of each model to produce per day)
- $I_w \geq 0$, for $w = 1,2,3$ (idle time at each workstation)

#### Objective

Minimize total idle time across all workstations:
$$
\min \; I_1 + I_2 + I_3
$$

#### Constraints

For each workstation $w = 1,2,3$:
- Idle time is effective capacity minus total processing time used:
$$
I_w = C_w - \sum_{k=1}^{101} t_{w,k} x_k
$$

- Idle time cannot be negative:
$$
I_w \geq 0
$$

- Production cannot use more than effective capacity:
$$
\sum_{k=1}^{101} t_{w,k} x_k \leq C_w
$$

- Nonnegativity and integrality:
$$
x_k \in \mathbb{Z}_{\geq 0}, \quad \forall k
$$

#### Data Table

The processing times $t_{w,k}$ are given by the following table (excerpt shown; full table as per CSV):

| Workstation | HiFi1_Minutes | HiFi2_Minutes | ... | HiFi101_Minutes | Maintenance_Percent |
|-------------|---------------|---------------|-----|-----------------|--------------------|
| 1           | 6             | 4             | ... | 9               | 10                 |
| 2           | 5             | 5             | ... | 3               | 14                 |
| 3           | 4             | 6             | ... | 6               | 12                 |

Where $t_{w,k}$ is the entry in row $w$, column "HiFi$k$_Minutes".

#### Complete Model

Minimize:
$$
I_1 + I_2 + I_3
$$

Subject to, for $w=1,2,3$:
$$
I_w = C_w - \sum_{k=1}^{101} t_{w,k} x_k
$$
$$
I_w \geq 0
$$
$$
x_k \in \mathbb{Z}_{\geq 0}, \quad \forall k=1,\ldots,101
$$

Where:
- $C_1 = 1,296$
- $C_2 = 1,238.4$
- $C_3 = 1,267.2$
- $t_{w,k}$ as per the retrieved CSV data above.