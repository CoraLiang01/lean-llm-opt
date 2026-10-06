Let $x_i$ be the production quantity of product $i$ ($i \in \{\text{P1}, \text{P2}, \ldots, \text{P111}\}$), which are nonnegative continuous variables.

Parameters:
- $p_i$: Unit_Profit of product $i$ (from unit_product_profits.csv)
- $a_{di}$: Processing time required by product $i$ on device $d$ (from device_time.csv)
- $c_d$: Monthly_Capacity of device $d$ (from monthly_device_capacity.csv)
- $d \in \{\text{A}, \text{B}, \text{C}, \text{D}, \text{E}, \text{F}, \text{G}, \text{H}, \text{I}, \text{J}\}$

Objective:
\[
\max \sum_{i=\text{P1}}^{\text{P111}} p_i x_i
\]

Subject to (for each device $d$):

\[
\sum_{i=\text{P1}}^{\text{P111}} a_{di} x_i \leq c_d \qquad \forall d \in \{\text{A}, \text{B}, \text{C}, \text{D}, \text{E}, \text{F}, \text{G}, \text{H}, \text{I}, \text{J}\}
\]

\[
x_i \geq 0 \qquad \forall i \in \{\text{P1}, \ldots, \text{P111}\}
\]

---

Numerical Data:

**unit_product_profits.csv**

| Product | Unit_Profit |
|---------|-------------|
| P1      | 28.55       |
| P2      | 12.78       |
| P3      | 45.21       |
| P4      | 18.92       |
| P5      | 33.47       |
| P6      | 8.64        |
| P7      | 25.88       |
| P8      | 40.15       |
| P9      | 14.39       |
| P10     | 37.62       |
| ...     | ...         |
| P111    | 9.99        |

**device_time.csv** (excerpt; all 10 devices and 111 products included)

| Device | P1  | P2  | ... | P111 |
|--------|-----|-----|-----|------|
| A      | 8.1 | 2.5 | ... | 8.5  |
| B      |10.5 | 5.2 | ... | 8.7  |
| C      | 2.1 |13.4 | ... | 2.9  |
| D      | 5.8 | 1.2 | ... | 8.3  |
| E      | 9.3 | 4.1 | ... | 6.7  |
| F      | 3.8 |14.2 | ... | 3.6  |
| G      | 7.2 | 2.8 | ... | 5.9  |
| H      |11.7 | 6.3 | ... | 1.4  |
| I      | 1.1 |11.3 | ... | 7.3  |
| J      | 4.6 | 0.2 | ... | 1.4  |

**monthly_device_capacity.csv**

| Device | Monthly_Capacity |
|--------|------------------|
| A      | 3500             |
| B      | 4200             |
| C      | 4500             |
| D      | 2800             |
| E      | 3300             |
| F      | 3800             |
| G      | 4100             |
| H      | 3900             |
| I      | 4800             |
| J      | 3100             |

---

Complete Model:

\[
\begin{align*}
\max\ & \sum_{i=\text{P1}}^{\text{P111}} p_i x_i \\
\text{s.t.}\quad
& \sum_{i=\text{P1}}^{\text{P111}} a_{A,i} x_i \leq 3500 \\
& \sum_{i=\text{P1}}^{\text{P111}} a_{B,i} x_i \leq 4200 \\
& \sum_{i=\text{P1}}^{\text{P111}} a_{C,i} x_i \leq 4500 \\
& \sum_{i=\text{P1}}^{\text{P111}} a_{D,i} x_i \leq 2800 \\
& \sum_{i=\text{P1}}^{\text{P111}} a_{E,i} x_i \leq 3300 \\
& \sum_{i=\text{P1}}^{\text{P111}} a_{F,i} x_i \leq 3800 \\
& \sum_{i=\text{P1}}^{\text{P111}} a_{G,i} x_i \leq 4100 \\
& \sum_{i=\text{P1}}^{\text{P111}} a_{H,i} x_i \leq 3900 \\
& \sum_{i=\text{P1}}^{\text{P111}} a_{I,i} x_i \leq 4800 \\
& \sum_{i=\text{P1}}^{\text{P111}} a_{J,i} x_i \leq 3100 \\
& x_i \geq 0 \qquad \forall i \in \{\text{P1}, \ldots, \text{P111}\}
\end{align*}
\]

Where:
- $p_i$ is the Unit_Profit for product $i$ (from unit_product_profits.csv)
- $a_{d,i}$ is the processing time required by product $i$ on device $d$ (from device_time.csv)
- The right-hand side of each constraint is the Monthly_Capacity for device $d$ (from monthly_device_capacity.csv)
- $x_i$ is the nonnegative continuous production quantity of product $i$