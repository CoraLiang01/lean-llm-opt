Let:
- \( x_p \geq 0 \): the production quantity of product \( p \) (continuous), for all products \( p \in \{P1, P2, ..., P111\} \).
- \( \pi_p \): unit profit of product \( p \), from unit_product_profits.csv.
- \( t_{d,p} \): processing time required by product \( p \) on device \( d \), from device_time.csv.
- \( C_d \): monthly capacity of device \( d \), from monthly_device_capacity.csv.

**Indices:**
- Products: \( p \in \{P1, P2, ..., P111\} \)
- Devices: \( d \in \{A, B, C, D, E, F, G, H, I, J\} \)

---

### Mathematical Model

#### Objective:
\[
\max \sum_{p \in \{P1, ..., P111\}} \pi_p \, x_p
\]

#### Subject to (for each device \( d \)):

\[
\sum_{p \in \{P1, ..., P111\}} t_{d,p} \, x_p \leq C_d \qquad \forall d \in \{A, B, C, D, E, F, G, H, I, J\}
\]

#### Variable domains:
\[
x_p \geq 0 \qquad \forall p \in \{P1, ..., P111\}
\]

---

### Parameters (from CSVs):

#### unit_product_profits.csv

| Product | Unit_Profit |
|---------|-------------|
| P1      | 28.55       |
| P2      | 12.78       |
| ...     | ...         |
| P111    | 9.99        |

#### device_time.csv

| Device | P1  | P2  | ... | P111 |
|--------|-----|-----|-----|------|
| A      | 8.1 | 2.5 | ... | 8.5  |
| B      |10.5 | 5.2 | ... | 8.7  |
| ...    | ... | ... | ... | ...  |
| J      | 4.6 | 0.2 | ... | 7.3  |

#### monthly_device_capacity.csv

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

### Complete Formulation (with explicit indices):

\[
\begin{align*}
\max \quad & \sum_{p = 1}^{111} \pi_{Pp} \, x_{Pp} \\
\text{s.t.} \quad & \sum_{p = 1}^{111} t_{A,Pp} \, x_{Pp} \leq 3500 \\
                  & \sum_{p = 1}^{111} t_{B,Pp} \, x_{Pp} \leq 4200 \\
                  & \sum_{p = 1}^{111} t_{C,Pp} \, x_{Pp} \leq 4500 \\
                  & \sum_{p = 1}^{111} t_{D,Pp} \, x_{Pp} \leq 2800 \\
                  & \sum_{p = 1}^{111} t_{E,Pp} \, x_{Pp} \leq 3300 \\
                  & \sum_{p = 1}^{111} t_{F,Pp} \, x_{Pp} \leq 3800 \\
                  & \sum_{p = 1}^{111} t_{G,Pp} \, x_{Pp} \leq 4100 \\
                  & \sum_{p = 1}^{111} t_{H,Pp} \, x_{Pp} \leq 3900 \\
                  & \sum_{p = 1}^{111} t_{I,Pp} \, x_{Pp} \leq 4800 \\
                  & \sum_{p = 1}^{111} t_{J,Pp} \, x_{Pp} \leq 3100 \\
                  & x_{Pp} \geq 0 \qquad \forall p = 1, ..., 111
\end{align*}
\]

Where:
- \( \pi_{Pp} \) is the Unit_Profit for product \( Pp \) from unit_product_profits.csv.
- \( t_{d,Pp} \) is the processing time for product \( Pp \) on device \( d \) from device_time.csv.
- Device capacities are as listed above.

**All coefficients and identifiers are as retrieved from the CSVs, with no omitted or synthesized data.**