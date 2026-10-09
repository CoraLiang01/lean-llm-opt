**Parameters:**

- Let $x_k \geq 0$ be the production quantity of product $k$, for $k \in \{\text{P1}, \ldots, \text{P111}\}$.
- Let $p_k$ be the unit profit of product $k$ (from unit_product_profits.csv).
- Let $a_{d,k}$ be the processing time required by product $k$ on device $d$ (from device_time.csv).
- Let $c_d$ be the monthly capacity of device $d$ (from monthly_device_capacity.csv).

**Objective:**

$$
\max \sum_{k \in \{\text{P1}, \ldots, \text{P111}\}} p_k \, x_k
$$

**Subject to:**

For each device $d \in \{\text{A}, \text{B}, \text{C}, \text{D}, \text{E}, \text{F}, \text{G}, \text{H}, \text{I}, \text{J}\}$:
$$
\sum_{k \in \{\text{P1}, \ldots, \text{P111}\}} a_{d,k} \, x_k \leq c_d
$$

For all $k \in \{\text{P1}, \ldots, \text{P111}\}$:
$$
x_k \geq 0
$$

**Data (as retrieved):**

- Devices:  
  - A, B, C, D, E, F, G, H, I, J

- Device capacities (monthly_device_capacity.csv):  
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
  - P1, P2, ..., P111

- Unit profits (unit_product_profits.csv):  
  - P1: 28.55  
  - P2: 12.78  
  - ...  
  - P111: 9.99  

- Device times (device_time.csv):  
  - For each device $d$ and product $k$, $a_{d,k}$ is the value in row with Device $d$ and column $k$.

**Full Model:**

Let $x_k \geq 0$ for all $k \in \{\text{P1}, ..., \text{P111}\}$.

$$
\max \sum_{k = 1}^{111} p_k \, x_k
$$

Subject to, for each device $d$:
$$
\sum_{k = 1}^{111} a_{d,k} \, x_k \leq c_d
$$

Where:
- $p_k$ is the unit profit of product $k$ (from unit_product_profits.csv)
- $a_{d,k}$ is the processing time of product $k$ on device $d$ (from device_time.csv)
- $c_d$ is the monthly capacity of device $d$ (from monthly_device_capacity.csv)
- $x_k \geq 0$ for all $k$

All indices, coefficients, and identifiers are as in the retrieved data.