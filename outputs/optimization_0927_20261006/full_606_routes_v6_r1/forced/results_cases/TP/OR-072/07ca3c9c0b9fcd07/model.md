Let $n=24$ be the number of time periods (hours), indexed by $t=1,2,\ldots,24$ in the order given in the data.

Let $x_t\geq0$ be the number of drivers and crew members who start work at the beginning of period $t$ (continuous or integer, as appropriate).

Let $r_t$ be the required number of drivers and crew members in period $t$, as given below.

Minimize the total number assigned:
$$
\min \sum_{t=1}^{24} x_t
$$

Subject to, for each period $t=1,2,\ldots,24$:
$$
\sum_{k=0}^{3} x_{(t-k-1)\bmod 24 + 1} \geq r_t
$$
where $x_{(t-k-1)\bmod 24 + 1}$ denotes the number starting in the current or previous 3 periods (with wrap-around for the 24-hour schedule).

Non-negativity:
$$
x_t \geq 0 \quad \forall t=1,\ldots,24
$$

Where the required numbers $r_t$ are:

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

All variables and constraints are indexed in the order of the data.

Retrieved Information:
[
  {"Shift":"1","Time":"0:00-1:00","Number Required":"20"},
  {"Shift":"2","Time":"1:00-2:00","Number Required":"18"},
  {"Shift":"3","Time":"2:00-3:00","Number Required":"15"},
  {"Shift":"4","Time":"3:00-4:00","Number Required":"15"},
  {"Shift":"5","Time":"4:00-5:00","Number Required":"20"},
  {"Shift":"6","Time":"5:00-6:00","Number Required":"30"},
  {"Shift":"7","Time":"6:00-7:00","Number Required":"60"},
  {"Shift":"8","Time":"7:00-8:00","Number Required":"70"},
  {"Shift":"9","Time":"8:00-9:00","Number Required":"50"},
  {"Shift":"10","Time":"9:00-10:00","Number Required":"55"},
  {"Shift":"11","Time":"10:00-11:00","Number Required":"65"},
  {"Shift":"12","Time":"11:00-12:00","Number Required":"75"},
  {"Shift":"13","Time":"12:00-13:00","Number Required":"80"},
  {"Shift":"14","Time":"13:00-14:00","Number Required":"70"},
  {"Shift":"15","Time":"14:00-15:00","Number Required":"60"},
  {"Shift":"16","Time":"15:00-16:00","Number Required":"55"},
  {"Shift":"17","Time":"16:00-17:00","Number Required":"60"},
  {"Shift":"18","Time":"17:00-18:00","Number Required":"75"},
  {"Shift":"19","Time":"18:00-19:00","Number Required":"85"},
  {"Shift":"20","Time":"19:00-20:00","Number Required":"70"},
  {"Shift":"21","Time":"20:00-21:00","Number Required":"50"},
  {"Shift":"22","Time":"21:00-22:00","Number Required":"40"},
  {"Shift":"23","Time":"22:00-23:00","Number Required":"35"},
  {"Shift":"24","Time":"23:00-0:00","Number Required":"25"}
]