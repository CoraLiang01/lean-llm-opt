Let $x_k$ be the number of units of radio model HiFi-$k$ ($k=1,\ldots,101$) to produce per day. Let $t_{jk}$ be the processing time (in minutes) required for one unit of model $k$ at workstation $j$ ($j=1,2,3$), as given in the table below. Each workstation $j$ has a total daily time of 1,440 minutes, but only a fraction is available for production after maintenance: $0.90$ for workstation 1, $0.86$ for workstation 2, and $0.88$ for workstation 3.

Define the idle time at workstation $j$ as:
$$
\text{Idle}_j = \text{EffectiveCapacity}_j - \sum_{k=1}^{101} t_{jk} x_k
$$
where
\[
\text{EffectiveCapacity}_1 = 1,440 \times 0.90 = 1,296 \\
\text{EffectiveCapacity}_2 = 1,440 \times 0.86 = 1,238.4 \\
\text{EffectiveCapacity}_3 = 1,440 \times 0.88 = 1,267.2
\]

The objective is to minimize total idle time:
\[
\min \sum_{j=1}^3 \text{Idle}_j = \sum_{j=1}^3 \left( \text{EffectiveCapacity}_j - \sum_{k=1}^{101} t_{jk} x_k \right)
\]
which is equivalent to maximizing total processing time used:
\[
\max \sum_{j=1}^3 \sum_{k=1}^{101} t_{jk} x_k
\]
subject to the constraints below.

#### Mathematical Model

**Variables:**
\[
x_k \in \mathbb{Z}_{\geq 0} \quad \forall k=1,\ldots,101
\]

**Parameters:**
- $t_{jk}$: Processing time (in minutes) for one unit of model $k$ at workstation $j$ (see table below).
- $\text{EffectiveCapacity}_j$: Effective daily capacity (in minutes) at workstation $j$ after maintenance.

**Objective:**
\[
\min \sum_{j=1}^3 \left( \text{EffectiveCapacity}_j - \sum_{k=1}^{101} t_{jk} x_k \right)
\]

**Constraints:**
\[
\sum_{k=1}^{101} t_{jk} x_k \leq \text{EffectiveCapacity}_j \qquad \forall j=1,2,3
\]
\[
x_k \geq 0 \text{ and integer} \qquad \forall k=1,\ldots,101
\]

**Data Table (source order):**

| Workstation | HiFi1_Minutes | HiFi2_Minutes | ... | HiFi101_Minutes | Maintenance_Percent |
|-------------|---------------|---------------|-----|-----------------|--------------------|
| 1           | 6             | 4             | ... | 9               | 10                 |
| 2           | 5             | 5             | ... | 3               | 14                 |
| 3           | 4             | 6             | ... | 6               | 12                 |

- $t_{1k}$ = value in row 1, column HiFi$k$_Minutes
- $t_{2k}$ = value in row 2, column HiFi$k$_Minutes
- $t_{3k}$ = value in row 3, column HiFi$k$_Minutes

**Effective capacities:**
- Workstation 1: $1,296$ minutes
- Workstation 2: $1,238.4$ minutes
- Workstation 3: $1,267.2$ minutes

**Decision variables:**
- $x_k$: number of units of HiFi-$k$ to produce per day, $k=1,\ldots,101$, integer and $\geq 0$

**Complete Model:**
\[
\begin{align*}
\min \quad & \sum_{j=1}^3 \left( \text{EffectiveCapacity}_j - \sum_{k=1}^{101} t_{jk} x_k \right) \\
\text{s.t.} \quad & \sum_{k=1}^{101} t_{jk} x_k \leq \text{EffectiveCapacity}_j \qquad \forall j=1,2,3 \\
& x_k \in \mathbb{Z}_{\geq 0} \qquad \forall k=1,\ldots,101
\end{align*}
\]
with all $t_{jk}$ and $\text{EffectiveCapacity}_j$ as specified above and in the retrieved data.