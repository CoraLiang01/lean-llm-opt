##### Decision Variables

Let $x_p \geq 0$ be the production quantity of product $p \in P$ (continuous).

##### Objective Function

\[
\max \sum_{p \in P} \pi_p x_p
\]

where $\pi_p$ is the unit profit of product $p$.

##### Constraints

1. **Device capacity constraints:** For each device $d \in D$,
   \[
   \sum_{p \in P} t_{d,p} x_p \leq C_d
   \]
   where $t_{d,p}$ is the processing time required by product $p$ on device $d$, and $C_d$ is the monthly capacity of device $d$.

2. **Nonnegativity:** For all $p \in P$,
   \[
   x_p \geq 0
   \]

---

#### Sets

- $P = \{\text{P1}, \text{P2}, \ldots, \text{P111}\}$ (products)
- $D = \{\text{A}, \text{B}, \text{C}, \text{D}, \text{E}, \text{F}, \text{G}, \text{H}, \text{I}, \text{J}\}$ (devices)

---

#### Parameters

**Unit Profits $\pi_p$ (from unit_product_profits.csv):**

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

(Full list: P1–P111, as retrieved above.)

**Device Processing Time Matrix $t_{d,p}$ (from device_time.csv):**

- For each device $d \in D$ and product $p \in P$, $t_{d,p}$ is given by the corresponding entry in device_time.csv.

Example (Device A):

| Product | P1  | P2  | P3  | ... | P111 |
|---------|-----|-----|-----|-----|------|
| A       | 8.1 | 2.5 |10.2 | ... | 8.5  |

(Repeat for devices B, C, ..., J. Each device row gives $t_{d,p}$ for all $p$.)

**Monthly Device Capacities $C_d$ (from monthly_device_capacity.csv):**

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

#### Complete Model

\[
\begin{align*}
\max\quad & \sum_{p \in P} \pi_p x_p \\
\text{s.t.}\quad & \sum_{p \in P} t_{d,p} x_p \leq C_d, \quad \forall d \in D \\
& x_p \geq 0, \quad \forall p \in P
\end{align*}
\]

where all sets, parameters, and coefficients are as listed above, with $t_{d,p}$, $\pi_p$, and $C_d$ taken directly from the provided CSV data.

---

##### Retrieved Information

- **Products $P$:** P1, P2, ..., P111
- **Devices $D$:** A, B, C, D, E, F, G, H, I, J
- **Unit Profits $\pi_p$:** as listed above for each product
- **Device Processing Times $t_{d,p}$:** full 10 (devices) × 111 (products) matrix from device_time.csv
- **Monthly Device Capacities $C_d$:** as listed above for each device

All vectors and matrices are to be used exactly as retrieved from the CSV files.