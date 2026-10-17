Let $x_p \geq 0$ be the continuous production quantity of product $p$ for each $p \in \{\text{P1}, \text{P2}, \ldots, \text{P111}\}$.

**Parameters:**

- $u_p$: Unit profit of product $p$ (from unit_product_profits.csv)
- $a_{dp}$: Processing time required by product $p$ on device $d$ (from device_time.csv)
- $c_d$: Monthly capacity of device $d$ (from monthly_device_capacity.csv)

**Sets:**

- Products: $p \in \{\text{P1}, \text{P2}, \ldots, \text{P111}\}$
- Devices: $d \in \{\text{A}, \text{B}, \text{C}, \text{D}, \text{E}, \text{F}, \text{G}, \text{H}, \text{I}, \text{J}\}$

---

**Objective:**

$$
\max \sum_{p} u_p \, x_p
$$

where the unit profits $u_p$ are:

\[
\begin{align*}
u_{\text{P1}} &= 28.55 \\
u_{\text{P2}} &= 12.78 \\
u_{\text{P3}} &= 45.21 \\
u_{\text{P4}} &= 18.92 \\
u_{\text{P5}} &= 33.47 \\
u_{\text{P6}} &= 8.64 \\
u_{\text{P7}} &= 25.88 \\
u_{\text{P8}} &= 40.15 \\
u_{\text{P9}} &= 14.39 \\
u_{\text{P10}} &= 37.62 \\
u_{\text{P11}} &= 10.25 \\
u_{\text{P12}} &= 29.81 \\
u_{\text{P13}} &= 48.77 \\
u_{\text{P14}} &= 16.53 \\
u_{\text{P15}} &= 42.09 \\
u_{\text{P16}} &= 9.11 \\
u_{\text{P17}} &= 22.46 \\
u_{\text{P18}} &= 36.73 \\
u_{\text{P19}} &= 11.99 \\
u_{\text{P20}} &= 31.28 \\
u_{\text{P21}} &= 7.45 \\
u_{\text{P22}} &= 27.6 \\
u_{\text{P23}} &= 43.91 \\
u_{\text{P24}} &= 15.82 \\
u_{\text{P25}} &= 39.36 \\
u_{\text{P26}} &= 6.23 \\
u_{\text{P27}} &= 20.71 \\
u_{\text{P28}} &= 34.98 \\
u_{\text{P29}} &= 13.24 \\
u_{\text{P30}} &= 32.53 \\
u_{\text{P31}} &= 5.1 \\
u_{\text{P32}} &= 24.18 \\
u_{\text{P33}} &= 47.85 \\
u_{\text{P34}} &= 19.3 \\
u_{\text{P35}} &= 41.44 \\
u_{\text{P36}} &= 11.02 \\
u_{\text{P37}} &= 26.57 \\
u_{\text{P38}} &= 38.2 \\
u_{\text{P39}} &= 17.76 \\
u_{\text{P40}} &= 46.12 \\
u_{\text{P41}} &= 9.89 \\
u_{\text{P42}} &= 21.34 \\
u_{\text{P43}} &= 33.86 \\
u_{\text{P44}} &= 14.88 \\
u_{\text{P45}} &= 30.41 \\
u_{\text{P46}} &= 6.5 \\
u_{\text{P47}} &= 28.93 \\
u_{\text{P48}} &= 49.2 \\
u_{\text{P49}} &= 18.15 \\
u_{\text{P50}} &= 44.78 \\
u_{\text{P51}} &= 10.57 \\
u_{\text{P52}} &= 23.69 \\
u_{\text{P53}} &= 35.51 \\
u_{\text{P54}} &= 16.03 \\
u_{\text{P55}} &= 38.83 \\
u_{\text{P56}} &= 5.88 \\
u_{\text{P57}} &= 29.45 \\
u_{\text{P58}} &= 42.33 \\
u_{\text{P59}} &= 12.41 \\
u_{\text{P60}} &= 37.06 \\
u_{\text{P61}} &= 7.99 \\
u_{\text{P62}} &= 21.9 \\
u_{\text{P63}} &= 46.99 \\
u_{\text{P64}} &= 19.87 \\
u_{\text{P65}} &= 40.7 \\
u_{\text{P66}} &= 8.34 \\
u_{\text{P67}} &= 26.11 \\
u_{\text{P68}} &= 39.54 \\
u_{\text{P69}} &= 14.07 \\
u_{\text{P70}} &= 35.79 \\
u_{\text{P71}} &= 6.92 \\
u_{\text{P72}} &= 23.03 \\
u_{\text{P73}} &= 45.56 \\
u_{\text{P74}} &= 17.2 \\
u_{\text{P75}} &= 43.27 \\
u_{\text{P76}} &= 9.52 \\
u_{\text{P77}} &= 28.08 \\
u_{\text{P78}} &= 41.1 \\
u_{\text{P79}} &= 15.46 \\
u_{\text{P80}} &= 34.3 \\
u_{\text{P81}} &= 5.43 \\
u_{\text{P82}} &= 20.25 \\
u_{\text{P83}} &= 48.38 \\
u_{\text{P84}} &= 18.68 \\
u_{\text{P85}} &= 40.01 \\
u_{\text{P86}} &= 11.45 \\
u_{\text{P87}} &= 25.32 \\
u_{\text{P88}} &= 37.58 \\
u_{\text{P89}} &= 13.62 \\
u_{\text{P90}} &= 31.84 \\
u_{\text{P91}} &= 7.27 \\
u_{\text{P92}} &= 24.75 \\
u_{\text{P93}} &= 49.88 \\
u_{\text{P94}} &= 16.85 \\
u_{\text{P95}} &= 42.82 \\
u_{\text{P96}} &= 10.13 \\
u_{\text{P97}} &= 27.36 \\
u_{\text{P98}} &= 36.19 \\
u_{\text{P99}} &= 12.8 \\
u_{\text{P100}} &= 30.09 \\
u_{\text{P101}} &= 6.07 \\
u_{\text{P102}} &= 22.67 \\
u_{\text{P103}} &= 44.42 \\
u_{\text{P104}} &= 19.54 \\
u_{\text{P105}} &= 47.41 \\
u_{\text{P106}} &= 8.78 \\
u_{\text{P107}} &= 29.97 \\
u_{\text{P108}} &= 38.96 \\
u_{\text{P109}} &= 15.11 \\
u_{\text{P110}} &= 33.05 \\
u_{\text{P111}} &= 9.99 \\
\end{align*}
\]

---

**Constraints:**

For each device $d \in \{\text{A}, \text{B}, \text{C}, \text{D}, \text{E}, \text{F}, \text{G}, \text{H}, \text{I}, \text{J}\}$:

$$
\sum_{p} a_{dp} \, x_p \leq c_d
$$

where the device capacities $c_d$ are:

\[
\begin{align*}
c_{\text{A}} &= 3500 \\
c_{\text{B}} &= 4200 \\
c_{\text{C}} &= 4500 \\
c_{\text{D}} &= 2800 \\
c_{\text{E}} &= 3300 \\
c_{\text{F}} &= 3800 \\
c_{\text{G}} &= 4100 \\
c_{\text{H}} &= 3900 \\
c_{\text{I}} &= 4800 \\
c_{\text{J}} &= 3100 \\
\end{align*}
\]

and the processing times $a_{dp}$ are given by the following table (from device_time.csv, source order):

| Device | P1   | P2   | ... | P111 |
|--------|------|------|-----|------|
|   A    | 8.1  | 2.5  | ... | 8.5  |
|   B    |10.5  | 5.2  | ... | 8.7  |
|   C    | 2.1  |13.4  | ... | 2.9  |
|   D    | 5.8  | 1.2  | ... | 8.3  |
|   E    | 9.3  | 4.1  | ... | 6.7  |
|   F    | 3.8  |14.2  | ... | 8.1  |
|   G    | 7.2  | 2.8  | ... | 5.9  |
|   H    |11.7  | 6.3  | ... | 1.4  |
|   I    | 1.1  |11.3  | ... | 7.3  |
|   J    | 4.6  | 0.2  | ... |12.4  |

(Full coefficients as in the retrieved device_time.csv, preserving source order.)

---

**Variable domains:**

$$
x_p \geq 0 \quad \text{(continuous)}, \quad \forall p \in \{\text{P1}, \ldots, \text{P111}\}
$$

---

**Complete Model:**

\[
\begin{align*}
\max \quad & \sum_{p=\text{P1}}^{\text{P111}} u_p \, x_p \\
\text{s.t.} \quad & \sum_{p=\text{P1}}^{\text{P111}} a_{dp} \, x_p \leq c_d, \quad \forall d \in \{\text{A}, \ldots, \text{J}\} \\
& x_p \geq 0, \quad \forall p \in \{\text{P1}, \ldots, \text{P111}\}
\end{align*}
\]

with all $u_p$, $a_{dp}$, and $c_d$ as specified above and in the retrieved data.