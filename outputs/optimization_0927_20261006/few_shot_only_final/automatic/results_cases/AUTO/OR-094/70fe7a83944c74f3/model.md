Let $x_m$ = number of units of radio model $m$ to produce per day, $m=1,\ldots,101$.

Let $a_{wm}$ = processing time (in minutes) required per unit of model $m$ at workstation $w$ (from the data below).

Let $C_1 = 1440 \times 0.90 = 1296$, $C_2 = 1440 \times 0.86 = 1238.4$, $C_3 = 1440 \times 0.88 = 1267.2$.

**Objective:**
\[
\min \sum_{w=1}^3 \left(C_w - \sum_{m=1}^{101} a_{wm} x_m\right)
\]
or equivalently,
\[
\max \sum_{w=1}^3 \sum_{m=1}^{101} a_{wm} x_m
\]

**Subject to:**

For each workstation $w=1,2,3$:
\[
\sum_{m=1}^{101} a_{wm} x_m \leq C_w
\]

For each model $m=1,\ldots,101$:
\[
x_m \in \mathbb{Z}_{\geq 0}
\]

**Parameters from workstation_times.csv:**

- Workstation 1 (maintenance percent: 10, $C_1=1296$):

| Model         | $a_{1m}$ (minutes) |
|---------------|-------------------|
| HiFi1         | 6                 |
| HiFi2         | 4                 |
| HiFi3         | 6                 |
| ...           | ...               |
| HiFi101       | 9                 |

- Workstation 2 (maintenance percent: 14, $C_2=1238.4$):

| Model         | $a_{2m}$ (minutes) |
|---------------|-------------------|
| HiFi1         | 5                 |
| HiFi2         | 5                 |
| HiFi3         | 5                 |
| ...           | ...               |
| HiFi101       | 3                 |

- Workstation 3 (maintenance percent: 12, $C_3=1267.2$):

| Model         | $a_{3m}$ (minutes) |
|---------------|-------------------|
| HiFi1         | 4                 |
| HiFi2         | 6                 |
| HiFi3         | 5                 |
| ...           | ...               |
| HiFi101       | 6                 |

**Explicitly, using the data as given:**

Let $x_1, x_2, \ldots, x_{101}$ be the integer number of units to produce of HiFi1, HiFi2, ..., HiFi101.

**Objective:**
\[
\min \left[1296 - \sum_{m=1}^{101} a_{1m} x_m + 1238.4 - \sum_{m=1}^{101} a_{2m} x_m + 1267.2 - \sum_{m=1}^{101} a_{3m} x_m\right]
\]

**Subject to:**
\[
\sum_{m=1}^{101} a_{1m} x_m \leq 1296
\]
\[
\sum_{m=1}^{101} a_{2m} x_m \leq 1238.4
\]
\[
\sum_{m=1}^{101} a_{3m} x_m \leq 1267.2
\]
\[
x_m \in \mathbb{Z}_{\geq 0}, \quad m=1,\ldots,101
\]

Where the coefficients $a_{wm}$ are as follows (from the CSV, in original order):

| Model   | $a_{1m}$ | $a_{2m}$ | $a_{3m}$ |
|---------|----------|----------|----------|
| HiFi1   | 6        | 5        | 4        |
| HiFi2   | 4        | 5        | 6        |
| HiFi3   | 6        | 5        | 5        |
| HiFi4   | 7        | 1        | 2        |
| HiFi5   | 6        | 7        | 6        |
| HiFi6   | 6        | 8        | 5        |
| HiFi7   | 8        | 7        | 3        |
| HiFi8   | 9        | 5        | 3        |
| HiFi9   | 6        | 6        | 4        |
| HiFi10  | 7        | 8        | 8        |
| HiFi11  | 1        | 9        | 6        |
| HiFi12  | 2        | 9        | 3        |
| HiFi13  | 4        | 2        | 3        |
| HiFi14  | 7        | 6        | 3        |
| HiFi15  | 3        | 9        | 7        |
| HiFi16  | 8        | 4        | 8        |
| HiFi17  | 3        | 1        | 3        |
| HiFi18  | 2        | 2        | 8        |
| HiFi19  | 4        | 9        | 1        |
| HiFi20  | 5        | 3        | 5        |
| HiFi21  | 8        | 8        | 3        |
| HiFi22  | 3        | 5        | 8        |
| HiFi23  | 2        | 9        | 5        |
| HiFi24  | 3        | 5        | 8        |
| HiFi25  | 9        | 8        | 4        |
| HiFi26  | 7        | 7        | 8        |
| HiFi27  | 3        | 1        | 6        |
| HiFi28  | 5        | 1        | 7        |
| HiFi29  | 7        | 9        | 9        |
| HiFi30  | 6        | 7        | 5        |
| HiFi31  | 2        | 1        | 3        |
| HiFi32  | 1        | 9        | 6        |
| HiFi33  | 5        | 6        | 3        |
| HiFi34  | 6        | 4        | 3        |
| HiFi35  | 5        | 7        | 3        |
| HiFi36  | 1        | 4        | 8        |
| HiFi37  | 7        | 8        | 4        |
| HiFi38  | 9        | 6        | 6        |
| HiFi39  | 8        | 5        | 3        |
| HiFi40  | 3        | 3        | 8        |
| HiFi41  | 3        | 6        | 3        |
| HiFi42  | 8        | 7        | 7        |
| HiFi43  | 2        | 6        | 5        |
| HiFi44  | 3        | 2        | 3        |
| HiFi45  | 3        | 1        | 1        |
| HiFi46  | 8        | 1        | 8        |
| HiFi47  | 9        | 3        | 9        |
| HiFi48  | 2        | 8        | 6        |
| HiFi49  | 3        | 4        | 6        |
| HiFi50  | 4        | 3        | 4        |
| HiFi51  | 2        | 6        | 7        |
| HiFi52  | 9        | 9        | 1        |
| HiFi53  | 2        | 8        | 9        |
| HiFi54  | 1        | 7        | 9        |
| HiFi55  | 8        | 2        | 3        |
| HiFi56  | 8        | 2        | 9        |
| HiFi57  | 4        | 5        | 6        |
| HiFi58  | 4        | 4        | 5        |
| HiFi59  | 6        | 3        | 7        |
| HiFi60  | 1        | 8        | 8        |
| HiFi61  | 6        | 8        | 9        |
| HiFi62  | 5        | 6        | 9        |
| HiFi63  | 3        | 6        | 8        |
| HiFi64  | 5        | 3        | 5        |
| HiFi65  | 1        | 1        | 4        |
| HiFi66  | 6        | 6        | 4        |
| HiFi67  | 6        | 2        | 3        |
| HiFi68  | 5        | 6        | 3        |
| HiFi69  | 3        | 1        | 8        |
| HiFi70  | 4        | 3        | 8        |
| HiFi71  | 3        | 7        | 2        |
| HiFi72  | 8        | 1        | 4        |
| HiFi73  | 1        | 1        | 9        |
| HiFi74  | 2        | 2        | 6        |
| HiFi75  | 3        | 8        | 7        |
| HiFi76  | 2        | 7        | 6        |
| HiFi77  | 8        | 8        | 7        |
| HiFi78  | 4        | 8        | 3        |
| HiFi79  | 4        | 7        | 1        |
| HiFi80  | 2        | 5        | 7        |
| HiFi81  | 7        | 2        | 6        |
| HiFi82  | 5        | 5        | 4        |
| HiFi83  | 1        | 6        | 3        |
| HiFi84  | 6        | 2        | 5        |
| HiFi85  | 4        | 3        | 7        |
| HiFi86  | 1        | 2        | 6        |
| HiFi87  | 3        | 3        | 3        |
| HiFi88  | 8        | 8        | 5        |
| HiFi89  | 3        | 4        | 2        |
| HiFi90  | 3        | 9        | 2        |
| HiFi91  | 3        | 6        | 9        |
| HiFi92  | 3        | 1        | 3        |
| HiFi93  | 6        | 4        | 6        |
| HiFi94  | 7        | 8        | 9        |
| HiFi95  | 6        | 8        | 7        |
| HiFi96  | 2        | 6        | 2        |
| HiFi97  | 1        | 8        | 4        |
| HiFi98  | 8        | 5        | 5        |
| HiFi99  | 9        | 5        | 8        |
| HiFi100 | 7        | 8        | 1        |
| HiFi101 | 9        | 3        | 6        |

**Variable domains:**
\[
x_m \in \mathbb{Z}_{\geq 0}, \quad m=1,\ldots,101
\]