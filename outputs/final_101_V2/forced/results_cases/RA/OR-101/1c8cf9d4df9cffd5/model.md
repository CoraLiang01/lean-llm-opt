Let $x_i$ denote the production quantity (in units) of product $i$ ($i \in \{\text{P1}, \text{P2}, \ldots, \text{P111}\}$). The variables $x_i$ are nonnegative and continuous.

Let $p_i$ be the unit profit of product $i$ (from unit_product_profits.csv).

Let $a_{di}$ be the processing time required by product $i$ on device $d$ (from device_time.csv, for $d \in \{\text{A},\ldots,\text{J}\}$).

Let $c_d$ be the monthly capacity of device $d$ (from monthly_device_capacity.csv).

The complete mathematical model is:

---

#### Decision Variables

$$
x_i \geq 0 \quad \text{(continuous)}, \quad \forall i \in \{\text{P1}, \ldots, \text{P111}\}
$$

#### Objective Function

$$
\max \sum_{i=\text{P1}}^{\text{P111}} p_i x_i
$$

where $p_i$ is as follows (source order):

\[
\begin{align*}
p_{\text{P1}} &= 28.55 \\
p_{\text{P2}} &= 12.78 \\
p_{\text{P3}} &= 45.21 \\
p_{\text{P4}} &= 18.92 \\
p_{\text{P5}} &= 33.47 \\
p_{\text{P6}} &= 8.64 \\
p_{\text{P7}} &= 25.88 \\
p_{\text{P8}} &= 40.15 \\
p_{\text{P9}} &= 14.39 \\
p_{\text{P10}} &= 37.62 \\
p_{\text{P11}} &= 10.25 \\
p_{\text{P12}} &= 29.81 \\
p_{\text{P13}} &= 48.77 \\
p_{\text{P14}} &= 16.53 \\
p_{\text{P15}} &= 42.09 \\
p_{\text{P16}} &= 9.11 \\
p_{\text{P17}} &= 22.46 \\
p_{\text{P18}} &= 36.73 \\
p_{\text{P19}} &= 11.99 \\
p_{\text{P20}} &= 31.28 \\
p_{\text{P21}} &= 7.45 \\
p_{\text{P22}} &= 27.6 \\
p_{\text{P23}} &= 43.91 \\
p_{\text{P24}} &= 15.82 \\
p_{\text{P25}} &= 39.36 \\
p_{\text{P26}} &= 6.23 \\
p_{\text{P27}} &= 20.71 \\
p_{\text{P28}} &= 34.98 \\
p_{\text{P29}} &= 13.24 \\
p_{\text{P30}} &= 32.53 \\
p_{\text{P31}} &= 5.1 \\
p_{\text{P32}} &= 24.18 \\
p_{\text{P33}} &= 47.85 \\
p_{\text{P34}} &= 19.3 \\
p_{\text{P35}} &= 41.44 \\
p_{\text{P36}} &= 11.02 \\
p_{\text{P37}} &= 26.57 \\
p_{\text{P38}} &= 38.2 \\
p_{\text{P39}} &= 17.76 \\
p_{\text{P40}} &= 46.12 \\
p_{\text{P41}} &= 9.89 \\
p_{\text{P42}} &= 21.34 \\
p_{\text{P43}} &= 33.86 \\
p_{\text{P44}} &= 14.88 \\
p_{\text{P45}} &= 30.41 \\
p_{\text{P46}} &= 6.5 \\
p_{\text{P47}} &= 28.93 \\
p_{\text{P48}} &= 49.2 \\
p_{\text{P49}} &= 18.15 \\
p_{\text{P50}} &= 44.78 \\
p_{\text{P51}} &= 10.57 \\
p_{\text{P52}} &= 23.69 \\
p_{\text{P53}} &= 35.51 \\
p_{\text{P54}} &= 16.03 \\
p_{\text{P55}} &= 38.83 \\
p_{\text{P56}} &= 5.88 \\
p_{\text{P57}} &= 29.45 \\
p_{\text{P58}} &= 42.33 \\
p_{\text{P59}} &= 12.41 \\
p_{\text{P60}} &= 37.06 \\
p_{\text{P61}} &= 7.99 \\
p_{\text{P62}} &= 21.9 \\
p_{\text{P63}} &= 46.99 \\
p_{\text{P64}} &= 19.87 \\
p_{\text{P65}} &= 40.7 \\
p_{\text{P66}} &= 8.34 \\
p_{\text{P67}} &= 26.11 \\
p_{\text{P68}} &= 39.54 \\
p_{\text{P69}} &= 14.07 \\
p_{\text{P70}} &= 35.79 \\
p_{\text{P71}} &= 6.92 \\
p_{\text{P72}} &= 23.03 \\
p_{\text{P73}} &= 45.56 \\
p_{\text{P74}} &= 17.2 \\
p_{\text{P75}} &= 43.27 \\
p_{\text{P76}} &= 9.52 \\
p_{\text{P77}} &= 28.08 \\
p_{\text{P78}} &= 41.1 \\
p_{\text{P79}} &= 15.46 \\
p_{\text{P80}} &= 34.3 \\
p_{\text{P81}} &= 5.43 \\
p_{\text{P82}} &= 20.25 \\
p_{\text{P83}} &= 48.38 \\
p_{\text{P84}} &= 18.68 \\
p_{\text{P85}} &= 40.01 \\
p_{\text{P86}} &= 11.45 \\
p_{\text{P87}} &= 25.32 \\
p_{\text{P88}} &= 37.58 \\
p_{\text{P89}} &= 13.62 \\
p_{\text{P90}} &= 31.84 \\
p_{\text{P91}} &= 7.27 \\
p_{\text{P92}} &= 24.75 \\
p_{\text{P93}} &= 49.88 \\
p_{\text{P94}} &= 16.85 \\
p_{\text{P95}} &= 42.82 \\
p_{\text{P96}} &= 10.13 \\
p_{\text{P97}} &= 27.36 \\
p_{\text{P98}} &= 36.19 \\
p_{\text{P99}} &= 12.8 \\
p_{\text{P100}} &= 30.09 \\
p_{\text{P101}} &= 6.07 \\
p_{\text{P102}} &= 22.67 \\
p_{\text{P103}} &= 44.42 \\
p_{\text{P104}} &= 19.54 \\
p_{\text{P105}} &= 47.41 \\
p_{\text{P106}} &= 8.78 \\
p_{\text{P107}} &= 29.97 \\
p_{\text{P108}} &= 38.96 \\
p_{\text{P109}} &= 15.11 \\
p_{\text{P110}} &= 33.05 \\
p_{\text{P111}} &= 9.99 \\
\end{align*}
\]

#### Constraints

For each device $d \in \{\text{A}, \text{B}, \text{C}, \text{D}, \text{E}, \text{F}, \text{G}, \text{H}, \text{I}, \text{J}\}$:

$$
\sum_{i=\text{P1}}^{\text{P111}} a_{di} x_i \leq c_d
$$

where $a_{di}$ is the processing time required by product $i$ on device $d$ (from device_time.csv, source order), and $c_d$ is the monthly capacity of device $d$ (from monthly_device_capacity.csv):

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

The coefficients $a_{di}$ are as given in device_time.csv, for each device $d$ and product $i$ (preserving the original file and row order).

#### Variable Domains

$$
x_i \geq 0 \quad \forall i \in \{\text{P1}, \ldots, \text{P111}\}
$$

---

This is a complete numerical formulation using all retrieved data, with all identifiers and coefficients preserved in source order.