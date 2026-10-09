**Indices:**
- Let $k$ index products, $k \in \{1,2,\ldots,111\}$, corresponding to Product IDs P1, P2, ..., P111.
- Let $d$ index devices, $d \in \{\text{A}, \text{B}, \text{C}, \text{D}, \text{E}, \text{F}, \text{G}, \text{H}, \text{I}, \text{J}\}$.

**Parameters:**
- $p_k$: Unit profit of product $k$ (from unit_product_profits.csv, column Unit_Profit, Product = P$k$).
- $a_{d,k}$: Processing time required on device $d$ per unit of product $k$ (from device_time.csv, Device = $d$, column P$k$).
- $c_d$: Monthly capacity of device $d$ (from monthly_device_capacity.csv, Device = $d$, column Monthly_Capacity).

**Decision variables:**
- $x_k \geq 0$: Production quantity of product $k$ (continuous).

**Objective:**
\[
\max \sum_{k=1}^{111} p_k x_k
\]

**Subject to:**

For each device $d \in \{\text{A}, \text{B}, \text{C}, \text{D}, \text{E}, \text{F}, \text{G}, \text{H}, \text{I}, \text{J}\}$:
\[
\sum_{k=1}^{111} a_{d,k} x_k \leq c_d
\]

For all $k = 1,\ldots,111$:
\[
x_k \geq 0
\]

**Explicitly, using the retrieved data:**

**Device capacities (monthly_device_capacity.csv):**
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

**Unit profits (unit_product_profits.csv):**
| Product | Unit_Profit |
|---------|-------------|
| P1      | 28.55       |
| P2      | 12.78       |
| ...     | ...         |
| P111    | 9.99        |

**Device times (device_time.csv):**
- For each Device $d$ (A–J), and each Product $k$ (P1–P111), $a_{d,k}$ is the value in row Device = $d$, column P$k$.

**Full model:**

\[
\begin{align*}
\max\quad & \sum_{k=1}^{111} p_k x_k \\
\text{s.t.}\quad & \sum_{k=1}^{111} a_{\text{A},k} x_k \leq 3500 \\
                 & \sum_{k=1}^{111} a_{\text{B},k} x_k \leq 4200 \\
                 & \sum_{k=1}^{111} a_{\text{C},k} x_k \leq 4500 \\
                 & \sum_{k=1}^{111} a_{\text{D},k} x_k \leq 2800 \\
                 & \sum_{k=1}^{111} a_{\text{E},k} x_k \leq 3300 \\
                 & \sum_{k=1}^{111} a_{\text{F},k} x_k \leq 3800 \\
                 & \sum_{k=1}^{111} a_{\text{G},k} x_k \leq 4100 \\
                 & \sum_{k=1}^{111} a_{\text{H},k} x_k \leq 3900 \\
                 & \sum_{k=1}^{111} a_{\text{I},k} x_k \leq 4800 \\
                 & \sum_{k=1}^{111} a_{\text{J},k} x_k \leq 3100 \\
                 & x_k \geq 0 \quad \forall k = 1,\ldots,111
\end{align*}
\]

Where:
- $p_k$ is the Unit_Profit for Product P$k$ from unit_product_profits.csv.
- $a_{d,k}$ is the processing time for Device $d$ and Product P$k$ from device_time.csv.
- $x_k$ is the continuous production quantity of Product P$k$.

**All coefficients and identifiers are as in the original data.**