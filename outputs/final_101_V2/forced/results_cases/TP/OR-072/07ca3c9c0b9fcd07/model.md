Let $x_t$ be the number of drivers and crew members assigned at the start of time period $t$ ($t=1,\ldots,24$), where each assignment covers 4 consecutive time periods (i.e., a 4-hour shift starting at period $t$).

Minimize the total number of drivers and crew members assigned:
$$
\min \sum_{t=1}^{24} x_t
$$

Subject to coverage constraints for each time period $s=1,\ldots,24$:
$$
\sum_{k=0}^{3} x_{(s-k-1 \bmod 24)+1} \geq r_s \quad \forall s=1,\ldots,24
$$
where $r_s$ is the required number of drivers and crew members in time period $s$ (from the data below), and indices wrap around modulo 24 (so, for example, for $s=2$, the sum is $x_2 + x_1 + x_{24} + x_{23}$).

Non-negativity and integrality:
$$
x_t \geq 0,\quad x_t \in \mathbb{Z},\quad \forall t=1,\ldots,24
$$

Where the required numbers $r_s$ are:

\[
\begin{array}{ll}
r_1 = 20 & \text{(0:00-1:00)} \\
r_2 = 18 & \text{(1:00-2:00)} \\
r_3 = 15 & \text{(2:00-3:00)} \\
r_4 = 15 & \text{(3:00-4:00)} \\
r_5 = 20 & \text{(4:00-5:00)} \\
r_6 = 30 & \text{(5:00-6:00)} \\
r_7 = 60 & \text{(6:00-7:00)} \\
r_8 = 70 & \text{(7:00-8:00)} \\
r_9 = 50 & \text{(8:00-9:00)} \\
r_{10} = 55 & \text{(9:00-10:00)} \\
r_{11} = 65 & \text{(10:00-11:00)} \\
r_{12} = 75 & \text{(11:00-12:00)} \\
r_{13} = 80 & \text{(12:00-13:00)} \\
r_{14} = 70 & \text{(13:00-14:00)} \\
r_{15} = 60 & \text{(14:00-15:00)} \\
r_{16} = 55 & \text{(15:00-16:00)} \\
r_{17} = 60 & \text{(16:00-17:00)} \\
r_{18} = 75 & \text{(17:00-18:00)} \\
r_{19} = 85 & \text{(18:00-19:00)} \\
r_{20} = 70 & \text{(19:00-20:00)} \\
r_{21} = 50 & \text{(20:00-21:00)} \\
r_{22} = 40 & \text{(21:00-22:00)} \\
r_{23} = 35 & \text{(22:00-23:00)} \\
r_{24} = 25 & \text{(23:00-0:00)} \\
\end{array}
\]

All variables and constraints are indexed in source order.