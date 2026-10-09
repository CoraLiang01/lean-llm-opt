##### Parameters

Let $T = 24$ be the number of time periods (hours in a day).

Let $r_t$ be the required number of drivers and crew members in period $t$ $(t=1,\ldots,24)$, as given below:

\[
\begin{array}{c|c|c}
\text{Shift} & \text{Time} & r_t \\
\hline
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

##### Decision Variables

Let $x_t \geq 0$ be the number of drivers and crew members who start work at the beginning of period $t$ $(t=1,\ldots,24)$.

##### Objective Function

\[
\min \sum_{t=1}^{24} x_t
\]

##### Constraints

Each driver/crew member works for 4 consecutive hours starting from their shift. For each period $t$, the total number of drivers and crew members on duty must be at least $r_t$. Thus, for each $t=1,\ldots,24$:

\[
x_t + x_{t-1} + x_{t-2} + x_{t-3} \geq r_t \qquad \forall t=1,\ldots,24
\]

where indices are taken modulo 24, i.e., $x_{t-k} = x_{(t-k-1 \bmod 24)+1}$ for $k=1,2,3$.

\[
x_t \geq 0 \qquad \forall t=1,\ldots,24
\]

##### Complete Model

\[
\begin{align*}
\min\ & \sum_{t=1}^{24} x_t \\
\text{s.t.}\quad & x_t + x_{t-1} + x_{t-2} + x_{t-3} \geq r_t \qquad \forall t=1,\ldots,24 \\
& x_t \geq 0 \qquad \forall t=1,\ldots,24
\end{align*}
\]

where $r_t$ is as given in the table above, and indices are modulo 24.

##### Retrieved Data

\[
\begin{array}{c|c|c}
\text{Shift} & \text{Time} & r_t \\
\hline
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