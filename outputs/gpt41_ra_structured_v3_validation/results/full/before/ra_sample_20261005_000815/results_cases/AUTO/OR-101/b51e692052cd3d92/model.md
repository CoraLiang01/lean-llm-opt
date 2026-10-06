**Decision Variables**

Let $x_p \geq 0$ denote the production quantity of product $p$ ($p \in \{\text{P1}, \text{P2}, \ldots, \text{P111}\}$). Variables are continuous and nonnegative.

---

**Parameters**

- $c_p$: Unit profit of product $p$ (from unit_product_profits.csv)
- $a_{d,p}$: Processing time required by product $p$ on device $d$ (from device_time.csv)
- $b_d$: Monthly operating capacity of device $d$ (from monthly_device_capacity.csv)

---

**Objective Function**

\[
\max \sum_{p \in \{\text{P1}, \ldots, \text{P111}\}} c_p \, x_p
\]

where the $c_p$ values are:

\[
\begin{aligned}
&c_{\text{P1}} = 28.55,\quad c_{\text{P2}} = 12.78,\quad c_{\text{P3}} = 45.21,\quad \ldots,\quad c_{\text{P111}} = 9.99
\end{aligned}
\]

---

**Constraints**

For each device $d \in \{\text{A}, \text{B}, \text{C}, \text{D}, \text{E}, \text{F}, \text{G}, \text{H}, \text{I}, \text{J}\}$:

\[
\sum_{p \in \{\text{P1}, \ldots, \text{P111}\}} a_{d,p} \, x_p \leq b_d
\]

where the $a_{d,p}$ values are as in device_time.csv, and $b_d$ values are:

\[
\begin{aligned}
&b_A = 3500 \\
&b_B = 4200 \\
&b_C = 4500 \\
&b_D = 2800 \\
&b_E = 3300 \\
&b_F = 3800 \\
&b_G = 4100 \\
&b_H = 3900 \\
&b_I = 4800 \\
&b_J = 3100 \\
\end{aligned}
\]

For example, for device A:

\[
8.1\,x_{\text{P1}} + 2.5\,x_{\text{P2}} + 10.2\,x_{\text{P3}} + \cdots + 8.5\,x_{\text{P111}} \leq 3500
\]

and similarly for each device, using the corresponding row from device_time.csv.

---

**Variable Domains**

\[
x_p \geq 0 \quad \text{(continuous)}, \quad \forall p \in \{\text{P1}, \ldots, \text{P111}\}
\]

---

**Complete Data Used**

- **unit_product_profits.csv**: All 111 products (P1–P111) and their Unit_Profit.
- **device_time.csv**: All 10 devices (A–J), all 111 products, and their processing times $a_{d,p}$.
- **monthly_device_capacity.csv**: All 10 devices (A–J) and their Monthly_Capacity $b_d$.

---

**Summary Mathematical Model**

\[
\begin{align*}
\max\ & \sum_{p = \text{P1}}^{\text{P111}} c_p\, x_p \\
\text{s.t.}\quad
& \sum_{p = \text{P1}}^{\text{P111}} a_{d,p}\, x_p \leq b_d, \quad \forall d \in \{\text{A},\ldots,\text{J}\} \\
& x_p \geq 0, \quad \forall p \in \{\text{P1},\ldots,\text{P111}\}
\end{align*}
\]

with all coefficients and identifiers as given in the retrieved CSVs.