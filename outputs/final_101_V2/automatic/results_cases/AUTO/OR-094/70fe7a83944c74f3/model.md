Let $x_k$ be the number of units of radio model HiFi-$k$ ($k=1,\ldots,101$) to produce per day. Let $t_{ik}$ be the processing time (in minutes) required for one unit of model $k$ at workstation $i$ ($i=1,2,3$), as given in the table below. Each workstation $i$ has a total daily time of 1,440 minutes, but only a fraction is available for production due to maintenance: 90% for workstation 1, 86% for workstation 2, and 88% for workstation 3.

Define the idle time at workstation $i$ as:
$$
\text{Idle}_i = \text{EffectiveCapacity}_i - \sum_{k=1}^{101} t_{ik} x_k
$$
where
\[
\text{EffectiveCapacity}_1 = 1440 \times 0.90 = 1296 \\
\text{EffectiveCapacity}_2 = 1440 \times 0.86 = 1238.4 \\
\text{EffectiveCapacity}_3 = 1440 \times 0.88 = 1267.2
\]

**Objective:**
\[
\min \sum_{i=1}^3 \text{Idle}_i = \sum_{i=1}^3 \left( \text{EffectiveCapacity}_i - \sum_{k=1}^{101} t_{ik} x_k \right)
\]
which is equivalent to
\[
\max \sum_{i=1}^3 \sum_{k=1}^{101} t_{ik} x_k
\]
subject to the constraints below.

**Constraints:**
\[
\sum_{k=1}^{101} t_{ik} x_k \leq \text{EffectiveCapacity}_i \qquad \forall i=1,2,3
\]
\[
x_k \in \mathbb{Z}_{\geq 0} \qquad \forall k=1,\ldots,101
\]

**Parameters from workstation_times.csv:**

| Workstation | HiFi1_Minutes | HiFi2_Minutes | ... | HiFi101_Minutes | Maintenance_Percent |
|-------------|--------------|--------------|-----|-----------------|--------------------|
| 1           | 6            | 4            | ... | 9               | 10                 |
| 2           | 5            | 5            | ... | 3               | 14                 |
| 3           | 4            | 6            | ... | 6               | 12                 |

- $t_{1,1} = 6$, $t_{1,2} = 4$, ..., $t_{1,101} = 9$
- $t_{2,1} = 5$, $t_{2,2} = 5$, ..., $t_{2,101} = 3$
- $t_{3,1} = 4$, $t_{3,2} = 6$, ..., $t_{3,101} = 6$

**Complete Model:**

\[
\begin{align*}
\min\ & \sum_{i=1}^3 \left( \text{EffectiveCapacity}_i - \sum_{k=1}^{101} t_{ik} x_k \right) \\
\text{s.t.}\quad
& \sum_{k=1}^{101} t_{ik} x_k \leq \text{EffectiveCapacity}_i \qquad \forall i=1,2,3 \\
& x_k \in \mathbb{Z}_{\geq 0} \qquad \forall k=1,\ldots,101 \\
\end{align*}
\]
where
\[
\text{EffectiveCapacity}_1 = 1296,\quad \text{EffectiveCapacity}_2 = 1238.4,\quad \text{EffectiveCapacity}_3 = 1267.2
\]
and all $t_{ik}$ are as given in the retrieved workstation_times.csv.