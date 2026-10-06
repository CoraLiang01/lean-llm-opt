Let $x_p$ denote the monthly production quantity of product $p$ ($p \in \{\text{P1}, \text{P2}, \ldots, \text{P111}\}$), where $x_p \geq 0$ and continuous.

Parameters (from the retrieved data):

- For each product $p$:
    - Unit profit: $u_p$ (from unit_product_profits.csv)
- For each device $d$:
    - Monthly capacity: $C_d$ (from monthly_device_capacity.csv)
    - Processing time per unit of product $p$ on device $d$: $a_{d,p}$ (from device_time.csv)

Indices:

- $p$: Product, $p \in \{\text{P1}, \ldots, \text{P111}\}$
- $d$: Device, $d \in \{\text{A}, \text{B}, \text{C}, \text{D}, \text{E}, \text{F}, \text{G}, \text{H}, \text{I}, \text{J}\}$

Objective:
\[
\max \sum_{p \in \{\text{P1}, \ldots, \text{P111}\}} u_p \, x_p
\]

Subject to, for each device $d$:
\[
\sum_{p \in \{\text{P1}, \ldots, \text{P111}\}} a_{d,p} \, x_p \leq C_d \qquad \forall d \in \{\text{A}, \ldots, \text{J}\}
\]

\[
x_p \geq 0 \qquad \forall p \in \{\text{P1}, \ldots, \text{P111}\}
\]

Where:

- $u_p$ is the Unit_Profit for product $p$ from unit_product_profits.csv (e.g., $u_{\text{P1}} = 28.55$, $u_{\text{P2}} = 12.78$, ..., $u_{\text{P111}} = 9.99$).
- $a_{d,p}$ is the processing time required by product $p$ on device $d$ from device_time.csv (e.g., $a_{\text{A},\text{P1}} = 8.1$, $a_{\text{B},\text{P1}} = 10.5$, ..., $a_{\text{J},\text{P111}} = 8.5$).
- $C_d$ is the Monthly_Capacity for device $d$ from monthly_device_capacity.csv (e.g., $C_{\text{A}} = 3500$, $C_{\text{B}} = 4200$, ..., $C_{\text{J}} = 3100$).

All coefficients and identifiers are as retrieved and must be used as given.