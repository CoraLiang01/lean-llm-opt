Let $x_k$ be the number of units of radio model HiFi-$k$ ($k=1,\ldots,101$) to produce per day. All $x_k$ are nonnegative integers.

Let $a_{ik}$ be the processing time (in minutes) required for one unit of model $k$ at workstation $i$ ($i=1,2,3$), as given in the data below.

Let $C_i$ be the effective daily capacity (in minutes) of workstation $i$ after maintenance.

**Data:**

- For all $i=1,2,3$, total available time per day: $1440$ minutes.
- Maintenance percentages: $10\%$ (WS1), $14\%$ (WS2), $12\%$ (WS3).
- Thus,
  - $C_1 = 1440 \times (1 - 0.10) = 1296$
  - $C_2 = 1440 \times (1 - 0.14) = 1238.4$
  - $C_3 = 1440 \times (1 - 0.12) = 1267.2$

- $a_{ik}$ values are from the table below (see "Processing Times" section).

---

### Mathematical Model

**Decision variables:**
- $x_k \in \mathbb{Z}_{\geq 0}$, for $k=1,\ldots,101$

**Auxiliary expressions:**
- Idle time at workstation $i$: $I_i = C_i - \sum_{k=1}^{101} a_{ik} x_k$, for $i=1,2,3$

**Objective:**
\[
\min \sum_{i=1}^3 I_i = \sum_{i=1}^3 \left( C_i - \sum_{k=1}^{101} a_{ik} x_k \right)
\]
which is equivalent to
\[
\max \sum_{i=1}^3 \sum_{k=1}^{101} a_{ik} x_k
\]
since $C_i$ are constants.

**Constraints:**
\[
\sum_{k=1}^{101} a_{ik} x_k \leq C_i \qquad \forall i=1,2,3
\]
\[
x_k \in \mathbb{Z}_{\geq 0} \qquad \forall k=1,\ldots,101
\]

---

#### Processing Times

Let $a_{ik}$ be as follows (from the CSV):

| Workstation | HiFi1 | HiFi2 | HiFi3 | ... | HiFi101 | Maintenance_Percent |
|-------------|-------|-------|-------|-----|---------|--------------------|
| 1           | 6     | 4     | 6     | ... | 9       | 10                 |
| 2           | 5     | 5     | 5     | ... | 3       | 14                 |
| 3           | 4     | 6     | 5     | ... | 6       | 12                 |

That is, for $k=1,\ldots,101$:
- $a_{1k}$ = value in "HiFi$k$_Minutes" for Workstation 1
- $a_{2k}$ = value in "HiFi$k$_Minutes" for Workstation 2
- $a_{3k}$ = value in "HiFi$k$_Minutes" for Workstation 3

---

### Complete Formulation

\[
\begin{align*}
\min\ & \sum_{i=1}^3 \left( C_i - \sum_{k=1}^{101} a_{ik} x_k \right) \\
\text{s.t.}\quad
& \sum_{k=1}^{101} a_{ik} x_k \leq C_i \qquad \forall i=1,2,3 \\
& x_k \in \mathbb{Z}_{\geq 0} \qquad \forall k=1,\ldots,101 \\
\end{align*}
\]

where:
- $C_1 = 1296$, $C_2 = 1238.4$, $C_3 = 1267.2$
- $a_{ik}$ as above.

**All coefficients and identifiers are as in the original data.**