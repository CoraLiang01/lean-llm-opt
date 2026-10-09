Let $x_k$ be the number of units of radio model $k$ (for $k = 1, \ldots, 101$) to produce per day. Let $Idle_j$ be the idle time (in minutes) at workstation $j$ ($j = 1,2,3$).

Let $a_{jk}$ be the processing time (in minutes) required at workstation $j$ for one unit of model $k$, as given in the table below.

The total available time per workstation per day is 1,440 minutes. The effective production time available after maintenance is:

- Workstation 1: $C_1 = 1,440 \times (1 - 0.10) = 1,296$ minutes
- Workstation 2: $C_2 = 1,440 \times (1 - 0.14) = 1,238.4$ minutes
- Workstation 3: $C_3 = 1,440 \times (1 - 0.12) = 1,267.2$ minutes

#### Parameters

- $a_{jk}$: Processing time (in minutes) for one unit of model $k$ at workstation $j$, from the table below.
- $C_j$: Effective daily capacity at workstation $j$ (as above).

#### Decision Variables

- $x_k \in \mathbb{Z}_{\geq 0}$: Number of units of model $k$ to produce per day.
- $Idle_j \geq 0$: Idle time (in minutes) at workstation $j$.

#### Objective

Minimize total idle time across all workstations:
$$
\min \sum_{j=1}^3 Idle_j
$$

#### Constraints

For each workstation $j = 1,2,3$:
1. Idle time definition:
   $$
   Idle_j = C_j - \sum_{k=1}^{101} a_{jk} x_k
   $$
2. Idle time nonnegativity:
   $$
   Idle_j \geq 0
   $$
3. Production cannot exceed effective capacity:
   $$
   \sum_{k=1}^{101} a_{jk} x_k \leq C_j
   $$

For all models $k = 1, \ldots, 101$:
4. Nonnegativity and integrality:
   $$
   x_k \in \mathbb{Z}_{\geq 0}
   $$

#### Data Table

The $a_{jk}$ coefficients are given by the following table (excerpt shown for first few models; all 101 must be included in the full model):

| Workstation | HiFi1_Minutes | HiFi2_Minutes | HiFi3_Minutes | ... | HiFi101_Minutes | Maintenance_Percent |
|-------------|---------------|---------------|---------------|-----|-----------------|--------------------|
| 1           | 6             | 4             | 6             | ... | 10              | 10                 |
| 2           | 5             | 5             | 5             | ... | 3               | 14                 |
| 3           | 4             | 6             | 5             | ... | 6               | 12                 |

Where $a_{1,1} = 6$, $a_{1,2} = 4$, ..., $a_{1,101} = 10$, $a_{2,1} = 5$, ..., $a_{3,101} = 6$, etc.

#### Complete Model

Minimize:
$$
Idle_1 + Idle_2 + Idle_3
$$

Subject to, for $j=1$:
$$
Idle_1 = 1,296 - \sum_{k=1}^{101} a_{1k} x_k \\
Idle_1 \geq 0 \\
\sum_{k=1}^{101} a_{1k} x_k \leq 1,296
$$

for $j=2$:
$$
Idle_2 = 1,238.4 - \sum_{k=1}^{101} a_{2k} x_k \\
Idle_2 \geq 0 \\
\sum_{k=1}^{101} a_{2k} x_k \leq 1,238.4
$$

for $j=3$:
$$
Idle_3 = 1,267.2 - \sum_{k=1}^{101} a_{3k} x_k \\
Idle_3 \geq 0 \\
\sum_{k=1}^{101} a_{3k} x_k \leq 1,267.2
$$

and for all $k=1,\ldots,101$:
$$
x_k \in \mathbb{Z}_{\geq 0}
$$

where all $a_{jk}$ are as given in the retrieved workstation_times.csv.