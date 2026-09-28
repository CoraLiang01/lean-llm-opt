##### Decision Variables

Let $x_m \in \mathbb{Z}_+$ be the number of units of radio model $m$ to produce per day, for $m \in M = \{\text{HiFi-1}, \text{HiFi-2}, \ldots, \text{HiFi-101}\}$.

##### Parameters

Let $t_{wm}$ be the processing time (in minutes) required at workstation $w$ for one unit of model $m$, for $w \in W = \{1,2,3\}$ and $m \in M$.

Workstation effective daily capacities:
- $C_1 = 1,296$ minutes
- $C_2 = 1,238.4$ minutes
- $C_3 = 1,267.2$ minutes

Processing times per unit (from workstation_times.csv):

| Model    | $t_{1m}$ | $t_{2m}$ | $t_{3m}$ |
|----------|----------|----------|----------|
| HiFi-1   | 6        | 5        | 4        |
| HiFi-2   | 4        | 5        | 6        |
| HiFi-3   | 6        | 5        | 5        |
| HiFi-4   | 7        | 1        | 2        |
| HiFi-5   | 6        | 7        | 6        |
| HiFi-6   | 6        | 8        | 5        |
| HiFi-7   | 8        | 7        | 3        |
| HiFi-8   | 9        | 5        | 3        |
| HiFi-9   | 6        | 6        | 4        |
| HiFi-10  | 7        | 8        | 8        |
| HiFi-11  | 1        | 9        | 6        |
| HiFi-12  | 2        | 9        | 3        |
| HiFi-13  | 4        | 2        | 3        |
| HiFi-14  | 7        | 6        | 3        |
| HiFi-15  | 3        | 9        | 7        |
| HiFi-16  | 8        | 4        | 8        |
| HiFi-17  | 3        | 1        | 3        |
| HiFi-18  | 2        | 2        | 8        |
| HiFi-19  | 4        | 9        | 1        |
| HiFi-20  | 5        | 3        | 5        |
| HiFi-21  | 8        | 8        | 3        |
| HiFi-22  | 3        | 5        | 8        |
| HiFi-23  | 2        | 9        | 5        |
| HiFi-24  | 3        | 5        | 8        |
| HiFi-25  | 9        | 8        | 4        |
| HiFi-26  | 7        | 7        | 8        |
| HiFi-27  | 3        | 1        | 6        |
| HiFi-28  | 5        | 1        | 7        |
| HiFi-29  | 7        | 9        | 9        |
| HiFi-30  | 6        | 7        | 5        |
| HiFi-31  | 2        | 1        | 3        |
| HiFi-32  | 1        | 9        | 6        |
| HiFi-33  | 5        | 6        | 3        |
| HiFi-34  | 6        | 4        | 3        |
| HiFi-35  | 5        | 7        | 3        |
| HiFi-36  | 1        | 4        | 8        |
| HiFi-37  | 7        | 8        | 4        |
| HiFi-38  | 9        | 6        | 6        |
| HiFi-39  | 8        | 5        | 3        |
| HiFi-40  | 3        | 3        | 8        |
| HiFi-41  | 3        | 6        | 3        |
| HiFi-42  | 8        | 7        | 7        |
| HiFi-43  | 2        | 6        | 5        |
| HiFi-44  | 3        | 2        | 3        |
| HiFi-45  | 3        | 1        | 1        |
| HiFi-46  | 8        | 1        | 8        |
| HiFi-47  | 9        | 3        | 9        |
| HiFi-48  | 2        | 8        | 6        |
| HiFi-49  | 3        | 4        | 6        |
| HiFi-50  | 4        | 3        | 4        |
| HiFi-51  | 2        | 6        | 7        |
| HiFi-52  | 9        | 9        | 1        |
| HiFi-53  | 2        | 8        | 9        |
| HiFi-54  | 1        | 7        | 9        |
| HiFi-55  | 8        | 2        | 3        |
| HiFi-56  | 8        | 2        | 9        |
| HiFi-57  | 4        | 5        | 6        |
| HiFi-58  | 4        | 4        | 5        |
| HiFi-59  | 6        | 3        | 7        |
| HiFi-60  | 1        | 8        | 8        |
| HiFi-61  | 6        | 8        | 9        |
| HiFi-62  | 5        | 6        | 9        |
| HiFi-63  | 3        | 6        | 8        |
| HiFi-64  | 5        | 3        | 5        |
| HiFi-65  | 1        | 1        | 4        |
| HiFi-66  | 6        | 6        | 4        |
| HiFi-67  | 6        | 2        | 3        |
| HiFi-68  | 5        | 6        | 3        |
| HiFi-69  | 3        | 1        | 8        |
| HiFi-70  | 4        | 3        | 8        |
| HiFi-71  | 3        | 7        | 2        |
| HiFi-72  | 8        | 1        | 4        |
| HiFi-73  | 1        | 1        | 9        |
| HiFi-74  | 2        | 2        | 6        |
| HiFi-75  | 3        | 8        | 7        |
| HiFi-76  | 2        | 7        | 6        |
| HiFi-77  | 8        | 8        | 7        |
| HiFi-78  | 4        | 8        | 3        |
| HiFi-79  | 4        | 7        | 1        |
| HiFi-80  | 2        | 5        | 7        |
| HiFi-81  | 7        | 2        | 6        |
| HiFi-82  | 5        | 5        | 4        |
| HiFi-83  | 1        | 6        | 3        |
| HiFi-84  | 6        | 2        | 5        |
| HiFi-85  | 4        | 3        | 7        |
| HiFi-86  | 1        | 2        | 6        |
| HiFi-87  | 3        | 3        | 3        |
| HiFi-88  | 8        | 8        | 5        |
| HiFi-89  | 3        | 4        | 2        |
| HiFi-90  | 3        | 9        | 2        |
| HiFi-91  | 3        | 6        | 9        |
| HiFi-92  | 3        | 1        | 3        |
| HiFi-93  | 6        | 4        | 6        |
| HiFi-94  | 7        | 8        | 9        |
| HiFi-95  | 6        | 8        | 7        |
| HiFi-96  | 2        | 6        | 2        |
| HiFi-97  | 1        | 8        | 4        |
| HiFi-98  | 8        | 5        | 5        |
| HiFi-99  | 9        | 5        | 8        |
| HiFi-100 | 7        | 8        | 1        |
| HiFi-101 | 9        | 3        | 6        |

##### Objective Function

Let the idle time at workstation $w$ be $I_w = C_w - \sum_{m \in M} t_{wm} x_m$.

Minimize total idle time:
$$
\min \sum_{w=1}^3 I_w = \sum_{w=1}^3 \left( C_w - \sum_{m \in M} t_{wm} x_m \right)
$$
which is equivalent to:
$$
\max \sum_{w=1}^3 \sum_{m \in M} t_{wm} x_m
$$
or, as originally stated,
$$
\min \left[ (C_1 + C_2 + C_3) - \sum_{w=1}^3 \sum_{m \in M} t_{wm} x_m \right]
$$

##### Constraints

1. Workstation time usage cannot exceed effective capacity:
   $$
   \sum_{m \in M} t_{wm} x_m \leq C_w, \quad \forall w \in \{1,2,3\}
   $$
   That is,
   - $\sum_{m \in M} t_{1m} x_m \leq 1,296$
   - $\sum_{m \in M} t_{2m} x_m \leq 1,238.4$
   - $\sum_{m \in M} t_{3m} x_m \leq 1,267.2$

2. Nonnegativity and integrality:
   $$
   x_m \in \mathbb{Z}_+, \quad \forall m \in M
   $$

##### Complete Mathematical Model

Let $M = \{\text{HiFi-1}, \ldots, \text{HiFi-101}\}$, $W = \{1,2,3\}$, $t_{wm}$ as above, and $C_1 = 1,296$, $C_2 = 1,238.4$, $C_3 = 1,267.2$.

$$
\begin{align*}
\min_{x_m \in \mathbb{Z}_+} \quad & (1,296 + 1,238.4 + 1,267.2) - \sum_{w=1}^3 \sum_{m \in M} t_{wm} x_m \\
\text{s.t.} \quad
& \sum_{m \in M} t_{1m} x_m \leq 1,296 \\
& \sum_{m \in M} t_{2m} x_m \leq 1,238.4 \\
& \sum_{m \in M} t_{3m} x_m \leq 1,267.2 \\
& x_m \in \mathbb{Z}_+, \quad \forall m \in M
\end{align*}
$$

All $t_{wm}$ values are as listed above for each model and workstation.