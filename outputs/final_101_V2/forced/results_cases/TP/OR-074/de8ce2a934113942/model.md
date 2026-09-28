Let the day be divided into $N=48$ half-hour periods, indexed by $t=1,2,\ldots,48$, in source order as in the data. Let $r_t$ be the required minimum number of waitstaff in period $t$ (from the Requirement column). Let $x_t$ be the number of waitstaff whose shift starts at period $t$ (decision variable). Each shift covers 16 consecutive periods (8 hours).

Decision variables:
$$
x_t \geq 0,\quad \text{integer},\quad t=1,\ldots,48
$$

Objective:
$$
\min \sum_{t=1}^{48} x_t
$$

Constraints (for each period $s=1,\ldots,48$):
$$
\sum_{k=0}^{15} x_{(s-k-1 \bmod 48)+1} \geq r_s
$$
where $x_{(s-k-1 \bmod 48)+1}$ is the number of staff whose shift started $k$ periods before period $s$ (with wrap-around for the 24-hour cycle).

Data (in source order):

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

Minimize total staff:
$$
\min \sum_{t=1}^{48} x_t
$$

Subject to, for each $s=1,\ldots,48$:
$$
\sum_{k=0}^{15} x_{(s-k-1 \bmod 48)+1} \geq r_s
$$

$$
x_t \geq 0,\quad \text{integer},\quad t=1,\ldots,48
$$

where $r_s$ is as listed above for each period $s$ in source order.