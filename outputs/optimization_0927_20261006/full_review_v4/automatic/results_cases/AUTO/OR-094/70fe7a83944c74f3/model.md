Let $x_m$ be the number of units of radio model $m$ (for $m = 1,2,\ldots,101$) to produce per day. All $x_m$ are nonnegative integers.

Let $t_{wm}$ be the processing time (in minutes) required for one unit of model $m$ at workstation $w$ ($w=1,2,3$), as given in the data below.

Let $C_w$ be the effective daily capacity (in minutes) of workstation $w$ after maintenance:
- $C_1 = 1440 \times (1 - 0.10) = 1296$
- $C_2 = 1440 \times (1 - 0.14) = 1238.4$
- $C_3 = 1440 \times (1 - 0.12) = 1267.2$

Let $I_w$ be the idle time at workstation $w$:
$$
I_w = C_w - \sum_{m=1}^{101} t_{wm} x_m
$$

Objective:
$$
\min \sum_{w=1}^3 I_w = \sum_{w=1}^3 \left( C_w - \sum_{m=1}^{101} t_{wm} x_m \right)
$$
which is equivalent to
$$
\max \sum_{w=1}^3 \sum_{m=1}^{101} t_{wm} x_m
$$
but as requested, we write the idle time minimization form.

Subject to:
\[
\sum_{m=1}^{101} t_{wm} x_m \leq C_w \qquad \forall w=1,2,3
\]
\[
x_m \in \mathbb{Z}_{\geq 0} \qquad \forall m=1,\ldots,101
\]

---

#### Data

**Workstation 1** (Maintenance_Percent: 10, $C_1 = 1296$):

| Model         | HiFi1 | HiFi2 | HiFi3 | HiFi4 | HiFi5 | HiFi6 | HiFi7 | HiFi8 | HiFi9 | HiFi10 | ... | HiFi101 |
|---------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|--------|-----|---------|
| Minutes/unit  | 6     | 4     | 6     | 7     | 6     | 6     | 8     | 9     | 6     | 7      | ... | 9       |

**Workstation 2** (Maintenance_Percent: 14, $C_2 = 1238.4$):

| Model         | HiFi1 | HiFi2 | HiFi3 | HiFi4 | HiFi5 | HiFi6 | HiFi7 | HiFi8 | HiFi9 | HiFi10 | ... | HiFi101 |
|---------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|--------|-----|---------|
| Minutes/unit  | 5     | 5     | 5     | 1     | 7     | 8     | 7     | 5     | 6     | 8      | ... | 3       |

**Workstation 3** (Maintenance_Percent: 12, $C_3 = 1267.2$):

| Model         | HiFi1 | HiFi2 | HiFi3 | HiFi4 | HiFi5 | HiFi6 | HiFi7 | HiFi8 | HiFi9 | HiFi10 | ... | HiFi101 |
|---------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|--------|-----|---------|
| Minutes/unit  | 4     | 6     | 5     | 2     | 6     | 5     | 3     | 3     | 4     | 8      | ... | 6       |

(Full per-model, per-workstation times as in the retrieved data.)

---

#### Complete Model

Let $x_m$ be the number of units of HiFi-$m$ to produce per day, $m=1,\ldots,101$.

Minimize total idle time:
\[
\min \left[ (1296 - \sum_{m=1}^{101} t_{1m} x_m) + (1238.4 - \sum_{m=1}^{101} t_{2m} x_m) + (1267.2 - \sum_{m=1}^{101} t_{3m} x_m) \right]
\]

Subject to:
\[
\sum_{m=1}^{101} t_{1m} x_m \leq 1296
\]
\[
\sum_{m=1}^{101} t_{2m} x_m \leq 1238.4
\]
\[
\sum_{m=1}^{101} t_{3m} x_m \leq 1267.2
\]
\[
x_m \in \mathbb{Z}_{\geq 0} \qquad \forall m=1,\ldots,101
\]

Where $t_{wm}$ are the per-unit processing times for model $m$ at workstation $w$ as given in the data above.