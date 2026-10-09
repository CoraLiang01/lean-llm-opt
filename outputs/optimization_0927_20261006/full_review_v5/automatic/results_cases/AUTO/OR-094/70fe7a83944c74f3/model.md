Let $x_k$ be the number of units of radio model HiFi$k$ ($k=1,\ldots,101$) to produce per day. All $x_k \in \mathbb{Z}_{\geq 0}$.

Let $t_{ik}$ be the processing time (in minutes) required for one unit of model $k$ at workstation $i$ ($i=1,2,3$), as given in the table below.

Let $C_i$ be the effective daily capacity (in minutes) of workstation $i$ after maintenance:
- $C_1 = 1440 \times (1 - 0.10) = 1296$
- $C_2 = 1440 \times (1 - 0.14) = 1238.4$
- $C_3 = 1440 \times (1 - 0.12) = 1267.2$

Let $Idle_i$ be the idle time at workstation $i$.

The model is:

Minimize total idle time:
$$
\min \sum_{i=1}^3 Idle_i
$$

Subject to, for each workstation $i=1,2,3$:
$$
\sum_{k=1}^{101} t_{ik} x_k + Idle_i = C_i
$$

$$
x_k \in \mathbb{Z}_{\geq 0} \quad \forall k=1,\ldots,101
$$
$$
Idle_i \geq 0 \quad \forall i=1,2,3
$$

---

#### Parameter Table: Processing Times (minutes per unit)

| Workstation | HiFi1 | HiFi2 | HiFi3 | HiFi4 | HiFi5 | HiFi6 | HiFi7 | HiFi8 | HiFi9 | HiFi10 | ... | HiFi101 | Maintenance_Percent |
|-------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|--------|-----|---------|---------------------|
| 1           | 6     | 4     | 6     | 7     | 6     | 6     | 8     | 9     | 6     | 7      | ... | 9       | 10                  |
| 2           | 5     | 5     | 5     | 1     | 7     | 8     | 7     | 5     | 6     | 8      | ... | 3       | 14                  |
| 3           | 4     | 6     | 5     | 2     | 6     | 5     | 3     | 3     | 4     | 8      | ... | 6       | 12                  |

(Full table includes all HiFi1 through HiFi101 columns as in the retrieved data.)

#### Effective Daily Capacities

| Workstation | $C_i$ (minutes) |
|-------------|-----------------|
| 1           | 1296            |
| 2           | 1238.4          |
| 3           | 1267.2          |

#### Decision Variables

- $x_k$: integer, number of units of HiFi$k$ to produce per day ($k=1,\ldots,101$)
- $Idle_i$: continuous, idle time at workstation $i$ ($i=1,2,3$)

#### Complete Model

Minimize:
$$
Idle_1 + Idle_2 + Idle_3
$$

Subject to:
\[
\begin{align*}
&\sum_{k=1}^{101} t_{1k} x_k + Idle_1 = 1296 \\
&\sum_{k=1}^{101} t_{2k} x_k + Idle_2 = 1238.4 \\
&\sum_{k=1}^{101} t_{3k} x_k + Idle_3 = 1267.2 \\
&x_k \in \mathbb{Z}_{\geq 0} \quad \forall k=1,\ldots,101 \\
&Idle_i \geq 0 \quad \forall i=1,2,3
\end{align*}
\]

All coefficients $t_{ik}$ are as given in the retrieved table above.