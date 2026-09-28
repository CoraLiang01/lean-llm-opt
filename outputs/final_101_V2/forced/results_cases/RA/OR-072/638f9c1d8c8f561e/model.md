Let $x_t$ be the number of drivers and crew members who start work at the beginning of time period $t$ ($t=1,\ldots,24$). Each assignment covers 4 consecutive periods (i.e., if a person starts at period $t$, they work during periods $t$, $t+1$, $t+2$, $t+3$, with periods taken modulo 24).

Let $r_t$ be the required number of drivers and crew members in period $t$, as given in the data.

Objective:
$$
\min \sum_{t=1}^{24} x_t
$$

Subject to, for each $t=1,\ldots,24$:
$$
\sum_{k=0}^{3} x_{(t-k-1 \bmod 24)+1} \geq r_t
$$

where $x_{(t-k-1 \bmod 24)+1}$ denotes the number of people who started in the current or previous 3 periods (with wrap-around from period 1 to 24).

Variable domains:
$$
x_t \in \mathbb{Z}_{\geq 0}, \quad \forall t=1,\ldots,24
$$

Where the required numbers $r_t$ are:

\[
\begin{array}{ll}
r_1 = 20 & r_{13} = 80 \\
r_2 = 18 & r_{14} = 70 \\
r_3 = 15 & r_{15} = 60 \\
r_4 = 15 & r_{16} = 55 \\
r_5 = 20 & r_{17} = 60 \\
r_6 = 30 & r_{18} = 75 \\
r_7 = 60 & r_{19} = 85 \\
r_8 = 70 & r_{20} = 70 \\
r_9 = 50 & r_{21} = 50 \\
r_{10} = 55 & r_{22} = 40 \\
r_{11} = 65 & r_{23} = 35 \\
r_{12} = 75 & r_{24} = 25 \\
\end{array}
\]

Complete model:

Minimize
$$
x_1 + x_2 + x_3 + \cdots + x_{24}
$$

Subject to (for $t=1$ to $24$):

\[
\begin{align*}
x_1 + x_{24} + x_{23} + x_{22} &\geq 20 \\
x_2 + x_1 + x_{24} + x_{23} &\geq 18 \\
x_3 + x_2 + x_1 + x_{24} &\geq 15 \\
x_4 + x_3 + x_2 + x_1 &\geq 15 \\
x_5 + x_4 + x_3 + x_2 &\geq 20 \\
x_6 + x_5 + x_4 + x_3 &\geq 30 \\
x_7 + x_6 + x_5 + x_4 &\geq 60 \\
x_8 + x_7 + x_6 + x_5 &\geq 70 \\
x_9 + x_8 + x_7 + x_6 &\geq 50 \\
x_{10} + x_9 + x_8 + x_7 &\geq 55 \\
x_{11} + x_{10} + x_9 + x_8 &\geq 65 \\
x_{12} + x_{11} + x_{10} + x_9 &\geq 75 \\
x_{13} + x_{12} + x_{11} + x_{10} &\geq 80 \\
x_{14} + x_{13} + x_{12} + x_{11} &\geq 70 \\
x_{15} + x_{14} + x_{13} + x_{12} &\geq 60 \\
x_{16} + x_{15} + x_{14} + x_{13} &\geq 55 \\
x_{17} + x_{16} + x_{15} + x_{14} &\geq 60 \\
x_{18} + x_{17} + x_{16} + x_{15} &\geq 75 \\
x_{19} + x_{18} + x_{17} + x_{16} &\geq 85 \\
x_{20} + x_{19} + x_{18} + x_{17} &\geq 70 \\
x_{21} + x_{20} + x_{19} + x_{18} &\geq 50 \\
x_{22} + x_{21} + x_{20} + x_{19} &\geq 40 \\
x_{23} + x_{22} + x_{21} + x_{20} &\geq 35 \\
x_{24} + x_{23} + x_{22} + x_{21} &\geq 25 \\
\end{align*}
\]

and

$$
x_t \in \mathbb{Z}_{\geq 0}, \quad t=1,\ldots,24
$$