##### Parameters

Let the day be divided into 48 half-hour intervals, indexed by $t = 1, 2, \ldots, 48$, corresponding to the following time slots and required minimum staff:

| $t$ | Time                   | Requirement $r_t$ |
|-----|------------------------|-------------------|
| 1   | 2:00am - 2:30am        | 2                 |
| 2   | 2:30am - 3:00am        | 3                 |
| 3   | 3:00am - 3:30am        | 4                 |
| 4   | 3:30am - 4:00am        | 6                 |
| 5   | 4:00am - 4:30am        | 5                 |
| 6   | 4:30am - 5:00am        | 4                 |
| 7   | 5:00am - 5:30am        | 5                 |
| 8   | 5:30am - 6:00am        | 6                 |
| 9   | 6:00am - 6:30am        | 7                 |
| 10  | 6:30am - 7:00am        | 8                 |
| 11  | 7:00am - 7:30am        | 9                 |
| 12  | 7:30am - 8:00am        | 9                 |
| 13  | 8:00am - 8:30am        | 8                 |
| 14  | 8:30am - 9:00am        | 8                 |
| 15  | 9:00am - 9:30am        | 9                 |
| 16  | 9:30am - 10:00am       | 9                 |
| 17  | 10:00am - 10:30am      | 10                |
| 18  | 10:30am - 11:00am      | 12                |
| 19  | 11:00am - 11:30am      | 11                |
| 20  | 11:30am - 12:00pm      | 11                |
| 21  | 12:00pm - 12:30pm      | 12                |
| 22  | 12:30pm - 1:00pm       | 11                |
| 23  | 1:00pm - 1:30pm        | 10                |
| 24  | 1:30pm - 2:00pm        | 9                 |
| 25  | 2:00pm - 2:30pm        | 8                 |
| 26  | 2:30pm - 3:00pm        | 7                 |
| 27  | 3:00pm - 3:30pm        | 6                 |
| 28  | 3:30pm - 4:00pm        | 5                 |
| 29  | 4:00pm - 4:30pm        | 5                 |
| 30  | 4:30pm - 5:00pm        | 6                 |
| 31  | 5:00pm - 5:30pm        | 7                 |
| 32  | 5:30pm - 6:00pm        | 8                 |
| 33  | 6:00pm - 6:30pm        | 9                 |
| 34  | 6:30pm - 7:00pm        | 10                |
| 35  | 7:00pm - 7:30pm        | 9                 |
| 36  | 7:30pm - 8:00pm        | 8                 |
| 37  | 8:00pm - 8:30pm        | 7                 |
| 38  | 8:30pm - 9:00pm        | 6                 |
| 39  | 9:00pm - 9:30pm        | 5                 |
| 40  | 9:30pm - 10:00pm       | 4                 |
| 41  | 10:00pm - 10:30pm      | 4                 |
| 42  | 10:30pm - 11:00pm      | 3                 |
| 43  | 11:00pm - 11:30pm      | 3                 |
| 44  | 11:30pm - 12:00am      | 3                 |
| 45  | 12:00am - 12:30am      | 3                 |
| 46  | 12:30am - 1:00am       | 4                 |
| 47  | 1:00am - 1:30am        | 4                 |
| 48  | 1:30am - 2:00am        | 4                 |

Let $r_t$ denote the required minimum number of waitstaff for time interval $t$.

##### Decision Variables

Let $x_s \in \mathbb{Z}_{\geq 0}$ be the number of waitstaff whose shift starts at time slot $s$, for $s = 1, 2, \ldots, 48$.

Each shift covers 8 consecutive half-hour intervals (i.e., 4 hours), so a shift starting at $s$ covers intervals $s, s+1, \ldots, s+15$ (since 8 hours = 16 half-hours), with wrap-around modulo 48.

##### Objective Function

\[
\min \sum_{s=1}^{48} x_s
\]

##### Constraints

For each time interval $t = 1, 2, \ldots, 48$:

\[
\sum_{k=0}^{15} x_{(t - k - 1 \bmod 48) + 1} \geq r_t
\]

where $x_{(t - k - 1 \bmod 48) + 1}$ denotes the number of staff whose shift started $k$ half-hours before $t$, wrapping around the 48 intervals.

##### Variable Domains

\[
x_s \in \mathbb{Z}_{\geq 0}, \quad \forall s = 1, \ldots, 48
\]

##### Parameters (from 44.csv)

\[
\begin{align*}
r_1 &= 2 \\
r_2 &= 3 \\
r_3 &= 4 \\
r_4 &= 6 \\
r_5 &= 5 \\
r_6 &= 4 \\
r_7 &= 5 \\
r_8 &= 6 \\
r_9 &= 7 \\
r_{10} &= 8 \\
r_{11} &= 9 \\
r_{12} &= 9 \\
r_{13} &= 8 \\
r_{14} &= 8 \\
r_{15} &= 9 \\
r_{16} &= 9 \\
r_{17} &= 10 \\
r_{18} &= 12 \\
r_{19} &= 11 \\
r_{20} &= 11 \\
r_{21} &= 12 \\
r_{22} &= 11 \\
r_{23} &= 10 \\
r_{24} &= 9 \\
r_{25} &= 8 \\
r_{26} &= 7 \\
r_{27} &= 6 \\
r_{28} &= 5 \\
r_{29} &= 5 \\
r_{30} &= 6 \\
r_{31} &= 7 \\
r_{32} &= 8 \\
r_{33} &= 9 \\
r_{34} &= 10 \\
r_{35} &= 9 \\
r_{36} &= 8 \\
r_{37} &= 7 \\
r_{38} &= 6 \\
r_{39} &= 5 \\
r_{40} &= 4 \\
r_{41} &= 4 \\
r_{42} &= 3 \\
r_{43} &= 3 \\
r_{44} &= 3 \\
r_{45} &= 3 \\
r_{46} &= 4 \\
r_{47} &= 4 \\
r_{48} &= 4 \\
\end{align*}
\]

##### Complete Model

\[
\begin{align*}
\min \quad & \sum_{s=1}^{48} x_s \\
\text{s.t.} \quad & \sum_{k=0}^{15} x_{(t - k - 1 \bmod 48) + 1} \geq r_t, \quad \forall t = 1, \ldots, 48 \\
& x_s \in \mathbb{Z}_{\geq 0}, \quad \forall s = 1, \ldots, 48
\end{align*}
\]

where $x_s$ is the number of waitstaff starting at time slot $s$, and $r_t$ is the required minimum staff for time slot $t$ as listed above.