Let $x_k$ be the number of units of radio model $k$ (for $k = 1, \ldots, 101$) to produce per day. Let $I_w$ be the idle time (in minutes) at workstation $w$ ($w = 1,2,3$).

Let $a_{wk}$ be the processing time (in minutes) required at workstation $w$ per unit of model $k$, as given in the table below.

The effective daily capacity at each workstation is:
- Workstation 1: $C_1 = 1440 \times (1 - 0.10) = 1296$ minutes
- Workstation 2: $C_2 = 1440 \times (1 - 0.14) = 1238.4$ minutes
- Workstation 3: $C_3 = 1440 \times (1 - 0.12) = 1267.2$ minutes

Indices:
- $w \in \{1,2,3\}$ (workstations)
- $k \in \{1,2,\ldots,101\}$ (radio models, HiFi-1 to HiFi-101)

Parameters (from workstation_times.csv):

\[
\begin{array}{l|cccccc}
\text{Model} & a_{1k} & a_{2k} & a_{3k} \\
\hline
\text{HiFi-1} & 6 & 5 & 4 \\
\text{HiFi-2} & 4 & 5 & 6 \\
\text{HiFi-3} & 6 & 5 & 5 \\
\text{HiFi-4} & 7 & 1 & 2 \\
\text{HiFi-5} & 6 & 7 & 6 \\
\text{HiFi-6} & 6 & 8 & 5 \\
\text{HiFi-7} & 8 & 7 & 3 \\
\text{HiFi-8} & 9 & 5 & 3 \\
\text{HiFi-9} & 6 & 6 & 4 \\
\text{HiFi-10} & 7 & 8 & 8 \\
\vdots & \vdots & \vdots & \vdots \\
\text{HiFi-101} & 9 & 3 & 6 \\
\end{array}
\]
(Full table as in workstation_times.csv, columns HiFi1_Minutes to HiFi101_Minutes, rows Workstation 1, 2, 3.)

Variables:
- $x_k \in \mathbb{Z}_{\geq 0}$, for all $k$ (number of units of model $k$ to produce per day)
- $I_w \geq 0$, for all $w$ (idle time at workstation $w$)

Objective:
\[
\min \sum_{w=1}^3 I_w
\]

Constraints:

For each workstation $w$:
\[
\sum_{k=1}^{101} a_{wk} x_k + I_w = C_w \qquad \forall w \in \{1,2,3\}
\]
where
\[
C_1 = 1296,\quad C_2 = 1238.4,\quad C_3 = 1267.2
\]

Variable domains:
\[
x_k \in \mathbb{Z}_{\geq 0} \qquad \forall k \in \{1,\ldots,101\}
\]
\[
I_w \geq 0 \qquad \forall w \in \{1,2,3\}
\]

All coefficients $a_{wk}$ are taken directly from the corresponding entries in workstation_times.csv, with $a_{1k}$ from row Workstation 1, $a_{2k}$ from row Workstation 2, $a_{3k}$ from row Workstation 3, and $k$ corresponding to HiFi-1 through HiFi-101 in the given column order.

This model determines the nonnegative integer number of units of each radio model to produce per day to minimize total idle production time across all three workstations, subject to effective daily capacity after maintenance.