Let $x_t$ be the number of drivers and crew members who start working at the beginning of time period $t$ ($t=1,2,\ldots,24$). Each person works for 4 consecutive hours (periods). The required number of drivers and crew members in each period is given by $r_t$.

Given data (in source order):

| Shift ($t$) | Time           | $r_t$ (Number Required) |
|-------------|----------------|------------------------|
| 1           | 0:00-1:00      | 20                     |
| 2           | 1:00-2:00      | 18                     |
| 3           | 2:00-3:00      | 15                     |
| 4           | 3:00-4:00      | 15                     |
| 5           | 4:00-5:00      | 20                     |
| 6           | 5:00-6:00      | 30                     |
| 7           | 6:00-7:00      | 60                     |
| 8           | 7:00-8:00      | 70                     |
| 9           | 8:00-9:00      | 50                     |
| 10          | 9:00-10:00     | 55                     |
| 11          | 10:00-11:00    | 65                     |
| 12          | 11:00-12:00    | 75                     |
| 13          | 12:00-13:00    | 80                     |
| 14          | 13:00-14:00    | 70                     |
| 15          | 14:00-15:00    | 60                     |
| 16          | 15:00-16:00    | 55                     |
| 17          | 16:00-17:00    | 60                     |
| 18          | 17:00-18:00    | 75                     |
| 19          | 18:00-19:00    | 85                     |
| 20          | 19:00-20:00    | 70                     |
| 21          | 20:00-21:00    | 50                     |
| 22          | 21:00-22:00    | 40                     |
| 23          | 22:00-23:00    | 35                     |
| 24          | 23:00-0:00     | 25                     |

Model:

Minimize the total number of drivers and crew members assigned:
$$
\min \sum_{t=1}^{24} x_t
$$

Subject to, for each time period $t=1,2,\ldots,24$:
$$
x_t + x_{t-1} + x_{t-2} + x_{t-3} \geq r_t
$$
where indices are taken modulo 24 (i.e., $x_0 = x_{24}$, $x_{-1} = x_{23}$, $x_{-2} = x_{22}$), to ensure coverage wraps around the 24-hour schedule.

Variable domains:
$$
x_t \in \mathbb{Z}_{\geq 0}, \quad t=1,2,\ldots,24
$$

Where:
- $x_t$ = number of drivers and crew members starting at the beginning of period $t$
- $r_t$ = required number of drivers and crew members in period $t$ (from the table above)