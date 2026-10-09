Let $x_m$ be the number of units of radio model $m$ (for $m = 1, 2, \ldots, 101$) to produce per day. Let $I_w$ be the idle time (in minutes) at workstation $w$ ($w = 1,2,3$).

Let $a_{w,m}$ be the processing time (in minutes) required at workstation $w$ for one unit of model $m$, as given in the data below.

The total available time per workstation per day is 1,440 minutes. The effective capacity at each workstation is:

- Workstation 1: $C_1 = 1,440 \times (1 - 0.10) = 1,296$ minutes
- Workstation 2: $C_2 = 1,440 \times (1 - 0.14) = 1,238.4$ minutes
- Workstation 3: $C_3 = 1,440 \times (1 - 0.12) = 1,267.2$ minutes

#### Decision Variables

- $x_m \in \mathbb{Z}_{\geq 0}$, for $m = 1, \ldots, 101$
- $I_w \geq 0$, for $w = 1,2,3$

#### Objective

Minimize total idle time across all workstations:
$$
\min \; I_1 + I_2 + I_3
$$

#### Constraints

For each workstation $w = 1,2,3$:
- Idle time is effective capacity minus total processing time used:
$$
I_w = C_w - \sum_{m=1}^{101} a_{w,m} x_m
$$

- Total processing time used cannot exceed effective capacity:
$$
\sum_{m=1}^{101} a_{w,m} x_m \leq C_w
$$

- Idle time is nonnegative:
$$
I_w \geq 0
$$

- Production variables are nonnegative integers:
$$
x_m \in \mathbb{Z}_{\geq 0}
$$

#### Data

Let the radio models be indexed as $m = 1, 2, \ldots, 101$ corresponding to HiFi1, HiFi2, ..., HiFi101.

The processing times $a_{w,m}$ are as follows (extracted from the CSV, in source order):

- For $w=1$ (Workstation 1, Maintenance_Percent: 10):

| Model         | $a_{1,m}$ (minutes) |
|---------------|---------------------|
| HiFi1         | 6                   |
| HiFi2         | 4                   |
| HiFi3         | 6                   |
| HiFi4         | 7                   |
| HiFi5         | 6                   |
| HiFi6         | 6                   |
| HiFi7         | 8                   |
| HiFi8         | 9                   |
| HiFi9         | 6                   |
| HiFi10        | 7                   |
| HiFi11        | 1                   |
| HiFi12        | 2                   |
| HiFi13        | 4                   |
| HiFi14        | 7                   |
| HiFi15        | 3                   |
| HiFi16        | 8                   |
| HiFi17        | 3                   |
| HiFi18        | 2                   |
| HiFi19        | 4                   |
| HiFi20        | 5                   |
| HiFi21        | 8                   |
| HiFi22        | 3                   |
| HiFi23        | 2                   |
| HiFi24        | 3                   |
| HiFi25        | 9                   |
| HiFi26        | 7                   |
| HiFi27        | 3                   |
| HiFi28        | 5                   |
| HiFi29        | 7                   |
| HiFi30        | 6                   |
| HiFi31        | 2                   |
| HiFi32        | 1                   |
| HiFi33        | 5                   |
| HiFi34        | 6                   |
| HiFi35        | 5                   |
| HiFi36        | 1                   |
| HiFi37        | 7                   |
| HiFi38        | 9                   |
| HiFi39        | 8                   |
| HiFi40        | 3                   |
| HiFi41        | 3                   |
| HiFi42        | 8                   |
| HiFi43        | 2                   |
| HiFi44        | 3                   |
| HiFi45        | 3                   |
| HiFi46        | 8                   |
| HiFi47        | 9                   |
| HiFi48        | 2                   |
| HiFi49        | 3                   |
| HiFi50        | 4                   |
| HiFi51        | 2                   |
| HiFi52        | 9                   |
| HiFi53        | 2                   |
| HiFi54        | 1                   |
| HiFi55        | 8                   |
| HiFi56        | 8                   |
| HiFi57        | 4                   |
| HiFi58        | 4                   |
| HiFi59        | 6                   |
| HiFi60        | 1                   |
| HiFi61        | 6                   |
| HiFi62        | 5                   |
| HiFi63        | 3                   |
| HiFi64        | 5                   |
| HiFi65        | 1                   |
| HiFi66        | 6                   |
| HiFi67        | 6                   |
| HiFi68        | 5                   |
| HiFi69        | 3                   |
| HiFi70        | 4                   |
| HiFi71        | 3                   |
| HiFi72        | 8                   |
| HiFi73        | 1                   |
| HiFi74        | 2                   |
| HiFi75        | 3                   |
| HiFi76        | 2                   |
| HiFi77        | 8                   |
| HiFi78        | 4                   |
| HiFi79        | 4                   |
| HiFi80        | 2                   |
| HiFi81        | 7                   |
| HiFi82        | 5                   |
| HiFi83        | 1                   |
| HiFi84        | 6                   |
| HiFi85        | 4                   |
| HiFi86        | 1                   |
| HiFi87        | 3                   |
| HiFi88        | 8                   |
| HiFi89        | 3                   |
| HiFi90        | 3                   |
| HiFi91        | 3                   |
| HiFi92        | 3                   |
| HiFi93        | 6                   |
| HiFi94        | 7                   |
| HiFi95        | 6                   |
| HiFi96        | 2                   |
| HiFi97        | 1                   |
| HiFi98        | 8                   |
| HiFi99        | 9                   |
| HiFi100       | 7                   |
| HiFi101       | 9                   |

- For $w=2$ (Workstation 2, Maintenance_Percent: 14):

| Model         | $a_{2,m}$ (minutes) |
|---------------|---------------------|
| HiFi1         | 5                   |
| HiFi2         | 5                   |
| HiFi3         | 5                   |
| HiFi4         | 1                   |
| HiFi5         | 7                   |
| HiFi6         | 8                   |
| HiFi7         | 7                   |
| HiFi8         | 5                   |
| HiFi9         | 6                   |
| HiFi10        | 8                   |
| HiFi11        | 9                   |
| HiFi12        | 9                   |
| HiFi13        | 2                   |
| HiFi14        | 6                   |
| HiFi15        | 9                   |
| HiFi16        | 4                   |
| HiFi17        | 1                   |
| HiFi18        | 2                   |
| HiFi19        | 9                   |
| HiFi20        | 3                   |
| HiFi21        | 8                   |
| HiFi22        | 5                   |
| HiFi23        | 9                   |
| HiFi24        | 5                   |
| HiFi25        | 8                   |
| HiFi26        | 7                   |
| HiFi27        | 1                   |
| HiFi28        | 1                   |
| HiFi29        | 9                   |
| HiFi30        | 7                   |
| HiFi31        | 1                   |
| HiFi32        | 9                   |
| HiFi33        | 6                   |
| HiFi34        | 4                   |
| HiFi35        | 7                   |
| HiFi36        | 4                   |
| HiFi37        | 8                   |
| HiFi38        | 6                   |
| HiFi39        | 5                   |
| HiFi40        | 3                   |
| HiFi41        | 6                   |
| HiFi42        | 7                   |
| HiFi43        | 6                   |
| HiFi44        | 2                   |
| HiFi45        | 1                   |
| HiFi46        | 1                   |
| HiFi47        | 3                   |
| HiFi48        | 8                   |
| HiFi49        | 4                   |
| HiFi50        | 3                   |
| HiFi51        | 6                   |
| HiFi52        | 9                   |
| HiFi53        | 8                   |
| HiFi54        | 7                   |
| HiFi55        | 2                   |
| HiFi56        | 2                   |
| HiFi57        | 5                   |
| HiFi58        | 4                   |
| HiFi59        | 3                   |
| HiFi60        | 8                   |
| HiFi61        | 8                   |
| HiFi62        | 6                   |
| HiFi63        | 6                   |
| HiFi64        | 3                   |
| HiFi65        | 1                   |
| HiFi66        | 6                   |
| HiFi67        | 2                   |
| HiFi68        | 6                   |
| HiFi69        | 1                   |
| HiFi70        | 3                   |
| HiFi71        | 7                   |
| HiFi72        | 1                   |
| HiFi73        | 1                   |
| HiFi74        | 2                   |
| HiFi75        | 8                   |
| HiFi76        | 7                   |
| HiFi77        | 8                   |
| HiFi78        | 8                   |
| HiFi79        | 7                   |
| HiFi80        | 5                   |
| HiFi81        | 2                   |
| HiFi82        | 5                   |
| HiFi83        | 6                   |
| HiFi84        | 2                   |
| HiFi85        | 3                   |
| HiFi86        | 2                   |
| HiFi87        | 3                   |
| HiFi88        | 8                   |
| HiFi89        | 4                   |
| HiFi90        | 9                   |
| HiFi91        | 6                   |
| HiFi92        | 1                   |
| HiFi93        | 4                   |
| HiFi94        | 8                   |
| HiFi95        | 8                   |
| HiFi96        | 6                   |
| HiFi97        | 8                   |
| HiFi98        | 5                   |
| HiFi99        | 5                   |
| HiFi100       | 8                   |
| HiFi101       | 3                   |

- For $w=3$ (Workstation 3, Maintenance_Percent: 12):

| Model         | $a_{3,m}$ (minutes) |
|---------------|---------------------|
| HiFi1         | 4                   |
| HiFi2         | 6                   |
| HiFi3         | 5                   |
| HiFi4         | 2                   |
| HiFi5         | 6                   |
| HiFi6         | 5                   |
| HiFi7         | 3                   |
| HiFi8         | 3                   |
| HiFi9         | 4                   |
| HiFi10        | 8                   |
| HiFi11        | 6                   |
| HiFi12        | 3                   |
| HiFi13        | 3                   |
| HiFi14        | 3                   |
| HiFi15        | 7                   |
| HiFi16        | 8                   |
| HiFi17        | 3                   |
| HiFi18        | 8                   |
| HiFi19        | 1                   |
| HiFi20        | 5                   |
| HiFi21        | 3                   |
| HiFi22        | 8                   |
| HiFi23        | 5                   |
| HiFi24        | 8                   |
| HiFi25        | 4                   |
| HiFi26        | 8                   |
| HiFi27        | 6                   |
| HiFi28        | 7                   |
| HiFi29        | 9                   |
| HiFi30        | 5                   |
| HiFi31        | 3                   |
| HiFi32        | 6                   |
| HiFi33        | 3                   |
| HiFi34        | 3                   |
| HiFi35        | 3                   |
| HiFi36        | 8                   |
| HiFi37        | 4                   |
| HiFi38        | 6                   |
| HiFi39        | 3                   |
| HiFi40        | 8                   |
| HiFi41        | 3                   |
| HiFi42        | 7                   |
| HiFi43        | 5                   |
| HiFi44        | 3                   |
| HiFi45        | 1                   |
| HiFi46        | 8                   |
| HiFi47        | 9                   |
| HiFi48        | 6                   |
| HiFi49        | 6                   |
| HiFi50        | 4                   |
| HiFi51        | 7                   |
| HiFi52        | 1                   |
| HiFi53        | 9                   |
| HiFi54        | 9                   |
| HiFi55        | 3                   |
| HiFi56        | 9                   |
| HiFi57        | 6                   |
| HiFi58        | 5                   |
| HiFi59        | 7                   |
| HiFi60        | 8                   |
| HiFi61        | 9                   |
| HiFi62        | 9                   |
| HiFi63        | 8                   |
| HiFi64        | 5                   |
| HiFi65        | 4                   |
| HiFi66        | 4                   |
| HiFi67        | 3                   |
| HiFi68        | 3                   |
| HiFi69        | 8                   |
| HiFi70        | 8                   |
| HiFi71        | 2                   |
| HiFi72        | 4                   |
| HiFi73        | 9                   |
| HiFi74        | 6                   |
| HiFi75        | 7                   |
| HiFi76        | 6                   |
| HiFi77        | 7                   |
| HiFi78        | 3                   |
| HiFi79        | 1                   |
| HiFi80        | 7                   |
| HiFi81        | 6                   |
| HiFi82        | 4                   |
| HiFi83        | 3                   |
| HiFi84        | 5                   |
| HiFi85        | 7                   |
| HiFi86        | 6                   |
| HiFi87        | 3                   |
| HiFi88        | 5                   |
| HiFi89        | 2                   |
| HiFi90        | 2                   |
| HiFi91        | 9                   |
| HiFi92        | 3                   |
| HiFi93        | 6                   |
| HiFi94        | 9                   |
| HiFi95        | 7                   |
| HiFi96        | 2                   |
| HiFi97        | 4                   |
| HiFi98        | 5                   |
| HiFi99        | 8                   |
| HiFi100       | 1                   |
| HiFi101       | 6                   |

#### Complete Model

Minimize:
$$
I_1 + I_2 + I_3
$$

Subject to, for $w=1$:
$$
I_1 = 1,296 - \sum_{m=1}^{101} a_{1,m} x_m \\
\sum_{m=1}^{101} a_{1,m} x_m \leq 1,296 \\
I_1 \geq 0
$$

for $w=2$:
$$
I_2 = 1,238.4 - \sum_{m=1}^{101} a_{2,m} x_m \\
\sum_{m=1}^{101} a_{2,m} x_m \leq 1,238.4 \\
I_2 \geq 0
$$

for $w=3$:
$$
I_3 = 1,267.2 - \sum_{m=1}^{101} a_{3,m} x_m \\
\sum_{m=1}^{101} a_{3,m} x_m \leq 1,267.2 \\
I_3 \geq 0
$$

and for all $m=1,\ldots,101$:
$$
x_m \in \mathbb{Z}_{\geq 0}
$$

where $a_{w,m}$ are as listed above, in the original CSV order.