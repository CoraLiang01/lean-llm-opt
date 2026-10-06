Let $x_k$ be the number of units of radio model HiFi-$k$ ($k=1,\ldots,101$) to produce per day. Let $t_{ik}$ be the processing time (in minutes) required for one unit of model $k$ at workstation $i$ ($i=1,2,3$), as given in the table below.

Let the effective daily capacity (in minutes) of workstation $i$ be $C_i = 1440 \times (1 - m_i/100)$, where $m_i$ is the maintenance percentage for workstation $i$.

Define the idle time at workstation $i$ as $Idle_i = C_i - \sum_{k=1}^{101} t_{ik} x_k$.

The objective is to minimize the total idle time across all workstations.

Minimize:
$$
\sum_{i=1}^3 Idle_i = \sum_{i=1}^3 \left[ C_i - \sum_{k=1}^{101} t_{ik} x_k \right]
$$
which is equivalent to:
$$
\min \left( \sum_{i=1}^3 C_i - \sum_{i=1}^3 \sum_{k=1}^{101} t_{ik} x_k \right)
$$
Since $\sum_{i=1}^3 C_i$ is constant, this is equivalent to:
$$
\max \sum_{i=1}^3 \sum_{k=1}^{101} t_{ik} x_k
$$

Subject to:
\[
\sum_{k=1}^{101} t_{ik} x_k \leq C_i \qquad \forall i=1,2,3
\]
\[
x_k \in \mathbb{Z}_{\geq 0} \qquad \forall k=1,\ldots,101
\]

Where:

- For workstation 1: $C_1 = 1440 \times 0.90 = 1296$ minutes
- For workstation 2: $C_2 = 1440 \times 0.86 = 1238.4$ minutes
- For workstation 3: $C_3 = 1440 \times 0.88 = 1267.2$ minutes

The processing times $t_{ik}$ are as follows (excerpt, full data as retrieved):

| Workstation | HiFi1_Minutes | HiFi2_Minutes | ... | HiFi101_Minutes | Maintenance_Percent |
|-------------|---------------|---------------|-----|-----------------|--------------------|
| 1           | 6             | 4             | ... | 9               | 10                 |
| 2           | 5             | 5             | ... | 3               | 14                 |
| 3           | 4             | 6             | ... | 6               | 12                 |

Decision variables:
- $x_k$: number of units of HiFi-$k$ to produce per day, $x_k \in \mathbb{Z}_{\geq 0}$ for $k=1,\ldots,101$

Parameters:
- $t_{ik}$: processing time (minutes) for one unit of HiFi-$k$ at workstation $i$ (from the table above)
- $C_1 = 1296$, $C_2 = 1238.4$, $C_3 = 1267.2$

Complete Model:

Minimize total idle time:
\[
\min \left[ (1296 - \sum_{k=1}^{101} t_{1k} x_k) + (1238.4 - \sum_{k=1}^{101} t_{2k} x_k) + (1267.2 - \sum_{k=1}^{101} t_{3k} x_k) \right]
\]

Subject to:
\[
\sum_{k=1}^{101} t_{1k} x_k \leq 1296
\]
\[
\sum_{k=1}^{101} t_{2k} x_k \leq 1238.4
\]
\[
\sum_{k=1}^{101} t_{3k} x_k \leq 1267.2
\]
\[
x_k \in \mathbb{Z}_{\geq 0} \qquad \forall k=1,\ldots,101
\]

Where all $t_{ik}$ are as given in the retrieved workstation_times.csv, preserving the original order and identifiers.