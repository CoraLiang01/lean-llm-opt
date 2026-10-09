Let $x_k \geq 0$ be the production quantity of product $k$ ($k \in \{\text{P1}, \ldots, \text{P111}\}$).

**Parameters:**
- $p_k$: Unit profit of product $k$ (from unit_product_profits.csv)
- $a_{ik}$: Processing time required by product $k$ on device $i$ (from device_time.csv)
- $c_i$: Monthly capacity of device $i$ (from monthly_device_capacity.csv)
- $i \in \{\text{A}, \text{B}, \text{C}, \text{D}, \text{E}, \text{F}, \text{G}, \text{H}, \text{I}, \text{J}\}$
- $k \in \{\text{P1}, \ldots, \text{P111}\}$

**Model:**

Objective:
$$
\max \sum_{k=\text{P1}}^{\text{P111}} p_k x_k
$$

Subject to, for each device $i$:
$$
\sum_{k=\text{P1}}^{\text{P111}} a_{ik} x_k \leq c_i \qquad \forall i \in \{\text{A}, \ldots, \text{J}\}
$$

Variable domains:
$$
x_k \geq 0 \qquad \forall k \in \{\text{P1}, \ldots, \text{P111}\}
$$

**Where:**

- $p_k$ is given by the "Unit_Profit" column in unit_product_profits.csv for each product $k$.
- $a_{ik}$ is given by the value in device_time.csv at row Device $i$, column $k$.
- $c_i$ is given by the "Monthly_Capacity" column in monthly_device_capacity.csv for each device $i$.

**Explicitly, using the retrieved data:**

- Devices:  
  A, B, C, D, E, F, G, H, I, J

- Device capacities:  
  - A: 3500  
  - B: 4200  
  - C: 4500  
  - D: 2800  
  - E: 3300  
  - F: 3800  
  - G: 4100  
  - H: 3900  
  - I: 4800  
  - J: 3100

- Products:  
  P1, P2, ..., P111

- For each $k$ (P1 to P111), $p_k$ is as in unit_product_profits.csv.

- For each $i$ (A to J), $a_{ik}$ is as in device_time.csv (row Device $i$, column $k$).

**Full model:**

$$
\begin{align*}
\max\quad & \sum_{k=\text{P1}}^{\text{P111}} p_k x_k \\
\text{s.t.}\quad & \sum_{k=\text{P1}}^{\text{P111}} a_{ik} x_k \leq c_i \qquad \forall i \in \{\text{A}, \ldots, \text{J}\} \\
& x_k \geq 0 \qquad \forall k \in \{\text{P1}, \ldots, \text{P111}\}
\end{align*}
$$

with all coefficients and identifiers as given in the source files.