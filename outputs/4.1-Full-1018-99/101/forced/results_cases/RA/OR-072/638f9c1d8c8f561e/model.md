Let $x_t$ be the number of drivers and crew members assigned to start work at the beginning of time period $t$ (Shift $t$), for $t = 1, 2, \ldots, 24$. Each person works continuously for 4 hours (i.e., covers 4 consecutive periods).

Let $r_t$ be the required number of drivers and crew members in time period $t$, as given in the data.

The model is:

Minimize the total number of drivers and crew members assigned:
$$
\min \sum_{t=1}^{24} x_t
$$

Subject to, for each time period $t = 1, 2, \ldots, 24$:
$$
x_t + x_{t-1} + x_{t-2} + x_{t-3} \geq r_t
$$
where indices are taken modulo 24 (i.e., $x_0 = x_{24}$, $x_{-1} = x_{23}$, $x_{-2} = x_{22}$, $x_{-3} = x_{21}$), to account for the 24-hour cycle.

With nonnegativity and integrality:
$$
x_t \in \mathbb{Z}_{\geq 0}, \quad \forall t = 1, 2, \ldots, 24
$$

Where the required numbers $r_t$ are:

\[
\begin{array}{lll}
t & \text{Time} & r_t \\
1 & 0:00-1:00 & 20 \\
2 & 1:00-2:00 & 18 \\
3 & 2:00-3:00 & 15 \\
4 & 3:00-4:00 & 15 \\
5 & 4:00-5:00 & 20 \\
6 & 5:00-6:00 & 30 \\
7 & 6:00-7:00 & 60 \\
8 & 7:00-8:00 & 70 \\
9 & 8:00-9:00 & 50 \\
10 & 9:00-10:00 & 55 \\
11 & 10:00-11:00 & 65 \\
12 & 11:00-12:00 & 75 \\
13 & 12:00-13:00 & 80 \\
14 & 13:00-14:00 & 70 \\
15 & 14:00-15:00 & 60 \\
16 & 15:00-16:00 & 55 \\
17 & 16:00-17:00 & 60 \\
18 & 17:00-18:00 & 75 \\
19 & 18:00-19:00 & 85 \\
20 & 19:00-20:00 & 70 \\
21 & 20:00-21:00 & 50 \\
22 & 21:00-22:00 & 40 \\
23 & 22:00-23:00 & 35 \\
24 & 23:00-0:00 & 25 \\
\end{array}
\]

So, the full model is:

Minimize
$$
x_1 + x_2 + \cdots + x_{24}
$$

Subject to, for $t = 1, \ldots, 24$:
$$
x_t + x_{t-1} + x_{t-2} + x_{t-3} \geq r_t
$$
with $x_t \in \mathbb{Z}_{\geq 0}$ for all $t$.

Where $r_t$ is as listed above, and indices wrap modulo 24.