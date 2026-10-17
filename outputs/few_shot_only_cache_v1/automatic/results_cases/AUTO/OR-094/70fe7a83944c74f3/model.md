Let $x_k$ be the number of units of radio model $k$ (for $k = 1, \ldots, 101$) to produce per day. All $x_k \in \mathbb{Z}_{\geq 0}$.

Let $t_{w,k}$ be the processing time (in minutes) required for one unit of model $k$ at workstation $w$ (for $w = 1,2,3$), as given in the table below.

Let $C_w$ be the effective daily capacity (in minutes) of workstation $w$ after maintenance:

- $C_1 = 1440 \times (1 - 0.10) = 1296$
- $C_2 = 1440 \times (1 - 0.14) = 1238.4$
- $C_3 = 1440 \times (1 - 0.12) = 1267.2$

Define the idle time at workstation $w$ as:
$$
\text{Idle}_w = C_w - \sum_{k=1}^{101} t_{w,k} x_k
$$

The objective is to minimize total idle time:
$$
\min \sum_{w=1}^3 \text{Idle}_w = \sum_{w=1}^3 \left( C_w - \sum_{k=1}^{101} t_{w,k} x_k \right)
$$
which is equivalent to:
$$
\max \sum_{w=1}^3 \sum_{k=1}^{101} t_{w,k} x_k
$$
subject to the constraints below.

#### Mathematical Model

**Decision variables:**
- $x_k \in \mathbb{Z}_{\geq 0}$, for $k = 1, \ldots, 101$

**Parameters:**
- $t_{w,k}$: processing time (in minutes) for model $k$ at workstation $w$ (see table below)
- $C_1 = 1296$, $C_2 = 1238.4$, $C_3 = 1267.2$

**Objective:**
\[
\min \sum_{w=1}^3 \left( C_w - \sum_{k=1}^{101} t_{w,k} x_k \right)
\]
or equivalently,
\[
\max \sum_{w=1}^3 \sum_{k=1}^{101} t_{w,k} x_k
\]

**Constraints:**
\[
\sum_{k=1}^{101} t_{w,k} x_k \leq C_w \qquad \forall w = 1,2,3
\]
\[
x_k \in \mathbb{Z}_{\geq 0} \qquad \forall k = 1, \ldots, 101
\]

**Parameter Table (partial, see CSV for full):**

| Workstation | HiFi1_Minutes | HiFi2_Minutes | ... | HiFi101_Minutes | Maintenance_Percent |
|-------------|---------------|---------------|-----|-----------------|--------------------|
| 1           | 6             | 4             | ... | 9               | 10                 |
| 2           | 5             | 5             | ... | 3               | 14                 |
| 3           | 4             | 6             | ... | 6               | 12                 |

Where $t_{w,k}$ is the value in row $w$, column "HiFi$k$_Minutes" of workstation_times.csv.

**Summary:**

- Decision variables: $x_k$ = units of HiFi-$k$ to produce per day, integer, $\geq 0$
- For each workstation $w$:
    - $\sum_{k=1}^{101} t_{w,k} x_k \leq C_w$
- Objective: minimize total idle time across all workstations

All coefficients and identifiers are as in the supplied CSV.