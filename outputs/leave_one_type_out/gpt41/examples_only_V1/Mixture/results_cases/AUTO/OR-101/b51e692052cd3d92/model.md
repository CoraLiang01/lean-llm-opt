Let $x_i$ denote the production quantity of product $i$ ($i \in \{\text{P1}, \text{P2}, \ldots, \text{P111}\}$), which is a nonnegative continuous variable.

Parameters:
- $p_i$: Unit profit of product $i$ (from unit_product_profits.csv)
- $a_{di}$: Processing time required by product $i$ on device $d$ (from device_time.csv)
- $c_d$: Monthly capacity of device $d$ (from monthly_device_capacity.csv), $d \in \{\text{A}, \text{B}, \ldots, \text{J}\}$

Objective:
\[
\max \sum_{i=\text{P1}}^{\text{P111}} p_i x_i
\]
where $p_i$ is as follows (partial list for illustration, full list from data):

\[
\begin{align*}
p_{\text{P1}} &= 28.55 \\
p_{\text{P2}} &= 12.78 \\
p_{\text{P3}} &= 45.21 \\
\vdots \\
p_{\text{P111}} &= 9.99 \\
\end{align*}
\]

Subject to, for each device $d$:
\[
\sum_{i=\text{P1}}^{\text{P111}} a_{di} x_i \leq c_d
\]
where $a_{di}$ and $c_d$ are as follows (partial list for illustration, full list from data):

Device capacities:
\[
\begin{align*}
c_A &= 3500 \\
c_B &= 4200 \\
c_C &= 4500 \\
c_D &= 2800 \\
c_E &= 3300 \\
c_F &= 3800 \\
c_G &= 4100 \\
c_H &= 3900 \\
c_I &= 4800 \\
c_J &= 3100 \\
\end{align*}
\]

For each device $d$ (A through J), the processing time coefficients $a_{di}$ are given by the corresponding row in device_time.csv, for each product $i$ (P1 through P111).

Variable domains:
\[
x_i \geq 0 \quad \forall i \in \{\text{P1}, \ldots, \text{P111}\}
\]

Complete Model:

\[
\begin{align*}
\max \quad & \sum_{i=\text{P1}}^{\text{P111}} p_i x_i \\
\text{s.t.} \quad & \sum_{i=\text{P1}}^{\text{P111}} a_{A,i} x_i \leq 3500 \\
                  & \sum_{i=\text{P1}}^{\text{P111}} a_{B,i} x_i \leq 4200 \\
                  & \sum_{i=\text{P1}}^{\text{P111}} a_{C,i} x_i \leq 4500 \\
                  & \sum_{i=\text{P1}}^{\text{P111}} a_{D,i} x_i \leq 2800 \\
                  & \sum_{i=\text{P1}}^{\text{P111}} a_{E,i} x_i \leq 3300 \\
                  & \sum_{i=\text{P1}}^{\text{P111}} a_{F,i} x_i \leq 3800 \\
                  & \sum_{i=\text{P1}}^{\text{P111}} a_{G,i} x_i \leq 4100 \\
                  & \sum_{i=\text{P1}}^{\text{P111}} a_{H,i} x_i \leq 3900 \\
                  & \sum_{i=\text{P1}}^{\text{P111}} a_{I,i} x_i \leq 4800 \\
                  & \sum_{i=\text{P1}}^{\text{P111}} a_{J,i} x_i \leq 3100 \\
                  & x_i \geq 0 \quad \forall i \in \{\text{P1}, \ldots, \text{P111}\}
\end{align*}
\]

Where:
- $p_i$ is the Unit_Profit for product $i$ from unit_product_profits.csv.
- $a_{d,i}$ is the processing time for product $i$ on device $d$ from device_time.csv.
- $c_d$ is the Monthly_Capacity for device $d$ from monthly_device_capacity.csv.

All coefficients and identifiers are as given in the retrieved data, in original file and row order.