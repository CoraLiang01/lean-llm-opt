Let $x_k$ be the number of units of radio model HiFi$k$ ($k=1,\ldots,101$) to produce per day. All $x_k \in \mathbb{Z}_{\geq 0}$.

Let $t_{jk}$ be the processing time (in minutes) required for one unit of model $k$ at workstation $j$ ($j=1,2,3$).

Let $C_j$ be the total available minutes per day at workstation $j$ (1,440 minutes), and $m_j$ the maintenance percentage at workstation $j$.

Let $E_j = C_j \cdot (1 - m_j/100)$ be the effective daily capacity at workstation $j$ after maintenance.

Let $I_j$ be the idle time at workstation $j$.

The data from workstation_times.csv is as follows:

- For $j=1$ (Workstation 1): $m_1 = 10$, $t_{1,1} = 6$, $t_{1,2} = 4$, ..., $t_{1,101} = 9$
- For $j=2$ (Workstation 2): $m_2 = 14$, $t_{2,1} = 5$, $t_{2,2} = 5$, ..., $t_{2,101} = 3$
- For $j=3$ (Workstation 3): $m_3 = 12$, $t_{3,1} = 4$, $t_{3,2} = 6$, ..., $t_{3,101} = 6$

The effective capacities are:
- $E_1 = 1440 \times 0.90 = 1296$
- $E_2 = 1440 \times 0.86 = 1238.4$
- $E_3 = 1440 \times 0.88 = 1267.2$

Define the idle time at each workstation:
$$
I_j = E_j - \sum_{k=1}^{101} t_{j,k} x_k \qquad \forall j=1,2,3
$$

The objective is to minimize total idle time:
$$
\min \sum_{j=1}^3 I_j
$$

Subject to:
\[
\begin{align*}
&\sum_{k=1}^{101} t_{j,k} x_k \leq E_j \qquad \forall j=1,2,3 \\
&x_k \in \mathbb{Z}_{\geq 0} \qquad \forall k=1,\ldots,101
\end{align*}
\]

Or, equivalently, substituting $I_j$:
\[
\min \sum_{j=1}^3 \left( E_j - \sum_{k=1}^{101} t_{j,k} x_k \right)
\]
which is equivalent to
\[
\max \sum_{j=1}^3 \sum_{k=1}^{101} t_{j,k} x_k
\]
subject to the same constraints.

But as requested, the model in terms of minimizing idle time is:

---

### Mathematical Model

**Parameters:**

- $t_{j,k}$: Processing time (minutes) for one unit of HiFi$k$ at workstation $j$ (from workstation_times.csv, see below)
- $E_1 = 1296$, $E_2 = 1238.4$, $E_3 = 1267.2$

**Decision variables:**

- $x_k \in \mathbb{Z}_{\geq 0}$: Number of units of HiFi$k$ to produce per day, for $k=1,\ldots,101$

**Objective:**
\[
\min \left[ (1296 - \sum_{k=1}^{101} t_{1,k} x_k) + (1238.4 - \sum_{k=1}^{101} t_{2,k} x_k) + (1267.2 - \sum_{k=1}^{101} t_{3,k} x_k) \right]
\]

**Subject to:**
\[
\begin{align*}
&\sum_{k=1}^{101} t_{1,k} x_k \leq 1296 \\
&\sum_{k=1}^{101} t_{2,k} x_k \leq 1238.4 \\
&\sum_{k=1}^{101} t_{3,k} x_k \leq 1267.2 \\
&x_k \in \mathbb{Z}_{\geq 0} \qquad \forall k=1,\ldots,101
\end{align*}
\]

**Where:**

- For $k=1,\ldots,101$ (HiFi1, HiFi2, ..., HiFi101):

| $k$ | $t_{1,k}$ | $t_{2,k}$ | $t_{3,k}$ |
|-----|-----------|-----------|-----------|
| 1   | 6         | 5         | 4         |
| 2   | 4         | 5         | 6         |
| 3   | 6         | 5         | 5         |
| 4   | 7         | 1         | 2         |
| 5   | 6         | 7         | 6         |
| ... | ...       | ...       | ...       |
| 101 | 9         | 3         | 6         |

(Use all values as retrieved from workstation_times.csv, in the original order.)

**Variable domains:**
\[
x_k \in \mathbb{Z}_{\geq 0} \qquad \forall k=1,\ldots,101
\]