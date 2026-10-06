Let $x_p \geq 0$ be the production quantity of product $p$ (continuous), for each product $p \in \{\text{P1}, \text{P2}, \ldots, \text{P111}\}$.

Let $d \in \{\text{A}, \text{B}, \text{C}, \text{D}, \text{E}, \text{F}, \text{G}, \text{H}, \text{I}, \text{J}\}$ index the 10 devices.

Let $t_{d,p}$ be the processing time required by product $p$ on device $d$ (from device_time.csv).

Let $c_d$ be the monthly capacity of device $d$ (from monthly_device_capacity.csv).

Let $u_p$ be the unit profit of product $p$ (from unit_product_profits.csv).

---

#### Objective:

$$
\max \sum_{p \in \{\text{P1},\ldots,\text{P111}\}} u_p \, x_p
$$

where the unit profits $u_p$ are:

\[
\begin{aligned}
&u_{\text{P1}} = 28.55,\quad u_{\text{P2}} = 12.78,\quad u_{\text{P3}} = 45.21,\quad u_{\text{P4}} = 18.92,\quad u_{\text{P5}} = 33.47,\\
&u_{\text{P6}} = 8.64,\quad u_{\text{P7}} = 25.88,\quad u_{\text{P8}} = 40.15,\quad u_{\text{P9}} = 14.39,\quad u_{\text{P10}} = 37.62,\\
&u_{\text{P11}} = 10.25,\quad u_{\text{P12}} = 29.81,\quad u_{\text{P13}} = 48.77,\quad u_{\text{P14}} = 16.53,\quad u_{\text{P15}} = 42.09,\\
&u_{\text{P16}} = 9.11,\quad u_{\text{P17}} = 22.46,\quad u_{\text{P18}} = 36.73,\quad u_{\text{P19}} = 11.99,\quad u_{\text{P20}} = 31.28,\\
&u_{\text{P21}} = 7.45,\quad u_{\text{P22}} = 27.60,\quad u_{\text{P23}} = 43.91,\quad u_{\text{P24}} = 15.82,\quad u_{\text{P25}} = 39.36,\\
&u_{\text{P26}} = 6.23,\quad u_{\text{P27}} = 20.71,\quad u_{\text{P28}} = 34.98,\quad u_{\text{P29}} = 13.24,\quad u_{\text{P30}} = 32.53,\\
&u_{\text{P31}} = 5.10,\quad u_{\text{P32}} = 24.18,\quad u_{\text{P33}} = 47.85,\quad u_{\text{P34}} = 19.30,\quad u_{\text{P35}} = 41.44,\\
&u_{\text{P36}} = 11.02,\quad u_{\text{P37}} = 26.57,\quad u_{\text{P38}} = 38.20,\quad u_{\text{P39}} = 17.76,\quad u_{\text{P40}} = 46.12,\\
&u_{\text{P41}} = 9.89,\quad u_{\text{P42}} = 21.34,\quad u_{\text{P43}} = 33.86,\quad u_{\text{P44}} = 14.88,\quad u_{\text{P45}} = 30.41,\\
&u_{\text{P46}} = 6.50,\quad u_{\text{P47}} = 28.93,\quad u_{\text{P48}} = 49.20,\quad u_{\text{P49}} = 18.15,\quad u_{\text{P50}} = 44.78,\\
&u_{\text{P51}} = 10.57,\quad u_{\text{P52}} = 23.69,\quad u_{\text{P53}} = 35.51,\quad u_{\text{P54}} = 16.03,\quad u_{\text{P55}} = 38.83,\\
&u_{\text{P56}} = 5.88,\quad u_{\text{P57}} = 29.45,\quad u_{\text{P58}} = 42.33,\quad u_{\text{P59}} = 12.41,\quad u_{\text{P60}} = 37.06,\\
&u_{\text{P61}} = 7.99,\quad u_{\text{P62}} = 21.90,\quad u_{\text{P63}} = 46.99,\quad u_{\text{P64}} = 19.87,\quad u_{\text{P65}} = 40.70,\\
&u_{\text{P66}} = 8.34,\quad u_{\text{P67}} = 26.11,\quad u_{\text{P68}} = 39.54,\quad u_{\text{P69}} = 14.07,\quad u_{\text{P70}} = 35.79,\\
&u_{\text{P71}} = 6.92,\quad u_{\text{P72}} = 23.03,\quad u_{\text{P73}} = 45.56,\quad u_{\text{P74}} = 17.20,\quad u_{\text{P75}} = 43.27,\\
&u_{\text{P76}} = 9.52,\quad u_{\text{P77}} = 28.08,\quad u_{\text{P78}} = 41.10,\quad u_{\text{P79}} = 15.46,\quad u_{\text{P80}} = 34.30,\\
&u_{\text{P81}} = 5.43,\quad u_{\text{P82}} = 20.25,\quad u_{\text{P83}} = 48.38,\quad u_{\text{P84}} = 18.68,\quad u_{\text{P85}} = 40.01,\\
&u_{\text{P86}} = 11.45,\quad u_{\text{P87}} = 25.32,\quad u_{\text{P88}} = 37.58,\quad u_{\text{P89}} = 13.62,\quad u_{\text{P90}} = 31.84,\\
&u_{\text{P91}} = 7.27,\quad u_{\text{P92}} = 24.75,\quad u_{\text{P93}} = 49.88,\quad u_{\text{P94}} = 16.85,\quad u_{\text{P95}} = 42.82,\\
&u_{\text{P96}} = 10.13,\quad u_{\text{P97}} = 27.36,\quad u_{\text{P98}} = 36.19,\quad u_{\text{P99}} = 12.80,\quad u_{\text{P100}} = 30.09,\\
&u_{\text{P101}} = 6.07,\quad u_{\text{P102}} = 22.67,\quad u_{\text{P103}} = 44.42,\quad u_{\text{P104}} = 19.54,\quad u_{\text{P105}} = 47.41,\\
&u_{\text{P106}} = 8.78,\quad u_{\text{P107}} = 29.97,\quad u_{\text{P108}} = 38.96,\quad u_{\text{P109}} = 15.11,\quad u_{\text{P110}} = 33.05,\\
&u_{\text{P111}} = 9.99
\end{aligned}
\]

---

#### Constraints:

For each device $d \in \{\text{A}, \text{B}, \text{C}, \text{D}, \text{E}, \text{F}, \text{G}, \text{H}, \text{I}, \text{J}\}$:

$$
\sum_{p \in \{\text{P1},\ldots,\text{P111}\}} t_{d,p} \, x_p \leq c_d
$$

where the device capacities $c_d$ are:

\[
\begin{aligned}
&c_{\text{A}} = 3500 \\
&c_{\text{B}} = 4200 \\
&c_{\text{C}} = 4500 \\
&c_{\text{D}} = 2800 \\
&c_{\text{E}} = 3300 \\
&c_{\text{F}} = 3800 \\
&c_{\text{G}} = 4100 \\
&c_{\text{H}} = 3900 \\
&c_{\text{I}} = 4800 \\
&c_{\text{J}} = 3100 \\
\end{aligned}
\]

and the processing times $t_{d,p}$ are as given in device_time.csv, e.g.:

- For device A: $t_{\text{A},\text{P1}} = 8.1$, $t_{\text{A},\text{P2}} = 2.5$, ..., $t_{\text{A},\text{P111}} = 8.5$
- For device B: $t_{\text{B},\text{P1}} = 10.5$, ..., $t_{\text{B},\text{P111}} = 8.7$
- ...
- For device J: $t_{\text{J},\text{P1}} = 4.6$, ..., $t_{\text{J},\text{P111}} = 12.4$

---

#### Variable domains:

$$
x_p \geq 0 \quad \forall p \in \{\text{P1},\ldots,\text{P111}\}
$$

---

#### Complete Model (Numerical Formulation):

\[
\begin{aligned}
\max \quad & \sum_{p = \text{P1}}^{\text{P111}} u_p \, x_p \\
\text{s.t.} \quad & \sum_{p = \text{P1}}^{\text{P111}} t_{d,p} \, x_p \leq c_d \quad \forall d \in \{\text{A},\ldots,\text{J}\} \\
& x_p \geq 0 \quad \forall p \in \{\text{P1},\ldots,\text{P111}\}
\end{aligned}
\]

where all $u_p$, $t_{d,p}$, and $c_d$ are as specified above and in the retrieved data.