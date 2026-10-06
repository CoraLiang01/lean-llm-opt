Let:
- $x_p$ = production quantity of product $p$ ($p \in \{\text{P1}, \text{P2}, \ldots, \text{P111}\}$), continuous and $\geq 0$.
- $u_p$ = unit profit of product $p$ (from unit_product_profits.csv).
- $t_{d,p}$ = processing time required by product $p$ on device $d$ (from device_time.csv).
- $C_d$ = monthly capacity of device $d$ (from monthly_device_capacity.csv), $d \in \{\text{A}, \text{B}, \ldots, \text{J}\}$.

The complete mathematical model is:

---

### Decision Variables

\[
x_p \geq 0 \quad \text{(continuous)}, \quad \forall p \in \{\text{P1}, \ldots, \text{P111}\}
\]

---

### Objective Function

\[
\max \sum_{p \in \{\text{P1}, \ldots, \text{P111}\}} u_p \cdot x_p
\]

where $u_p$ is the unit profit for product $p$ as given in unit_product_profits.csv.

---

### Constraints

#### Device Capacity Constraints

For each device $d \in \{\text{A}, \ldots, \text{J}\}$:

\[
\sum_{p \in \{\text{P1}, \ldots, \text{P111}\}} t_{d,p} \cdot x_p \leq C_d
\]

where:
- $t_{d,p}$ is the processing time required by product $p$ on device $d$ (from device_time.csv).
- $C_d$ is the monthly capacity of device $d$ (from monthly_device_capacity.csv).

That is, for each device:

- For device A: $\sum_{p} t_{\text{A},p} x_p \leq 3500$
- For device B: $\sum_{p} t_{\text{B},p} x_p \leq 4200$
- For device C: $\sum_{p} t_{\text{C},p} x_p \leq 4500$
- For device D: $\sum_{p} t_{\text{D},p} x_p \leq 2800$
- For device E: $\sum_{p} t_{\text{E},p} x_p \leq 3300$
- For device F: $\sum_{p} t_{\text{F},p} x_p \leq 3800$
- For device G: $\sum_{p} t_{\text{G},p} x_p \leq 4100$
- For device H: $\sum_{p} t_{\text{H},p} x_p \leq 3900$
- For device I: $\sum_{p} t_{\text{I},p} x_p \leq 4800$
- For device J: $\sum_{p} t_{\text{J},p} x_p \leq 3100$

---

### Variable Domains

\[
x_p \geq 0 \quad \text{(continuous)}, \quad \forall p \in \{\text{P1}, \ldots, \text{P111}\}
\]

---

### Data Mapping

- $u_p$ is from unit_product_profits.csv, column Unit_Profit, indexed by Product.
- $t_{d,p}$ is from device_time.csv, row Device $d$, column $p$.
- $C_d$ is from monthly_device_capacity.csv, column Monthly_Capacity, indexed by Device.

---

### Full Model (Expanded)

\[
\begin{align*}
\max \quad & \sum_{p=\text{P1}}^{\text{P111}} u_p x_p \\
\text{s.t.} \quad & \sum_{p=\text{P1}}^{\text{P111}} t_{\text{A},p} x_p \leq 3500 \\
& \sum_{p=\text{P1}}^{\text{P111}} t_{\text{B},p} x_p \leq 4200 \\
& \sum_{p=\text{P1}}^{\text{P111}} t_{\text{C},p} x_p \leq 4500 \\
& \sum_{p=\text{P1}}^{\text{P111}} t_{\text{D},p} x_p \leq 2800 \\
& \sum_{p=\text{P1}}^{\text{P111}} t_{\text{E},p} x_p \leq 3300 \\
& \sum_{p=\text{P1}}^{\text{P111}} t_{\text{F},p} x_p \leq 3800 \\
& \sum_{p=\text{P1}}^{\text{P111}} t_{\text{G},p} x_p \leq 4100 \\
& \sum_{p=\text{P1}}^{\text{P111}} t_{\text{H},p} x_p \leq 3900 \\
& \sum_{p=\text{P1}}^{\text{P111}} t_{\text{I},p} x_p \leq 4800 \\
& \sum_{p=\text{P1}}^{\text{P111}} t_{\text{J},p} x_p \leq 3100 \\
& x_p \geq 0 \quad \forall p \in \{\text{P1}, \ldots, \text{P111}\}
\end{align*}
\]

All coefficients and indices are to be taken directly from the provided CSV files, preserving their order and identifiers.