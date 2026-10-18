Let $x_t$ be the number of waitstaff who start their 8-hour shift at time interval $t$, where $t$ indexes the 48 half-hour intervals in the day, in the order given in the data.

Let $r_t$ be the required minimum number of waitstaff in interval $t$, as given in the "Requirement" column.

Define the intervals in order as follows (with $t=1$ corresponding to "2:00am - 2:30am", $t=2$ to "2:30am - 3:00am", ..., $t=48$ to "1:30am - 2:00am").

The model is:

Minimize total number of waitstaff scheduled:
$$
\min \sum_{t=1}^{48} x_t
$$

Subject to coverage constraints for each interval $s=1,\ldots,48$:
$$
\sum_{k=0}^{15} x_{(s-k-1 \bmod 48) + 1} \geq r_s, \quad \forall s=1,\ldots,48
$$

where $x_{(s-k-1 \bmod 48) + 1}$ denotes the number of staff who started in the $k$th previous interval (modulo 48, to wrap around midnight), and each staff covers 16 consecutive half-hour intervals (8 hours).

Variable domains:
$$
x_t \in \mathbb{Z}_{\geq 0}, \quad \forall t=1,\ldots,48
$$

Where the requirements $r_s$ are:

\[
\begin{array}{ll}
t & r_t \\
1 & 2 \\
2 & 3 \\
3 & 4 \\
4 & 6 \\
5 & 5 \\
6 & 4 \\
7 & 5 \\
8 & 6 \\
9 & 7 \\
10 & 8 \\
11 & 9 \\
12 & 9 \\
13 & 8 \\
14 & 8 \\
15 & 9 \\
16 & 9 \\
17 & 10 \\
18 & 12 \\
19 & 11 \\
20 & 11 \\
21 & 12 \\
22 & 11 \\
23 & 10 \\
24 & 9 \\
25 & 8 \\
26 & 7 \\
27 & 6 \\
28 & 5 \\
29 & 5 \\
30 & 6 \\
31 & 7 \\
32 & 8 \\
33 & 9 \\
34 & 10 \\
35 & 9 \\
36 & 8 \\
37 & 7 \\
38 & 6 \\
39 & 5 \\
40 & 4 \\
41 & 4 \\
42 & 3 \\
43 & 3 \\
44 & 3 \\
45 & 3 \\
46 & 4 \\
47 & 4 \\
48 & 4 \\
\end{array}
\]

Summary:

- $x_t$ = number of staff starting at interval $t$ ($t=1,\ldots,48$; see above for mapping to time intervals)
- Each staff covers 16 consecutive intervals (8 hours)
- For each interval $s$, the sum of $x_t$ for all $t$ whose shift covers $s$ must be at least $r_s$
- All $x_t$ are nonnegative integers

Minimize $\sum_{t=1}^{48} x_t$ subject to the above constraints.