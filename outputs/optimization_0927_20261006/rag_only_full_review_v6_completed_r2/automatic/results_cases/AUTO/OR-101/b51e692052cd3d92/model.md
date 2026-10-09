Let \( x_p \geq 0 \) be the continuous production quantity of product \( p \) (for \( p \) in \(\{P1, P2, ..., P111\}\)).

Parameters:
- Let \( \pi_p \) be the unit profit of product \( p \) (from unit_product_profits.csv).
- Let \( t_{d,p} \) be the processing time required by product \( p \) on device \( d \) (from device_time.csv, for \( d \) in \(\{A,B,C,D,E,F,G,H,I,J\}\)).
- Let \( C_d \) be the monthly capacity of device \( d \) (from monthly_device_capacity.csv).

Objective:
\[
\max \sum_{p=1}^{111} \pi_{Pp} \cdot x_{Pp}
\]
where the products and their profits are, in source order:
\[
\begin{align*}
&\pi_{P1} = 28.55,\ \pi_{P2} = 12.78,\ \pi_{P3} = 45.21,\ \pi_{P4} = 18.92,\ \pi_{P5} = 33.47,\\
&\pi_{P6} = 8.64,\ \pi_{P7} = 25.88,\ \pi_{P8} = 40.15,\ \pi_{P9} = 14.39,\ \pi_{P10} = 37.62,\\
&\ldots\\
&\pi_{P111} = 9.99
\end{align*}
\]
(see full list in the evidence above).

Subject to, for each device \( d \) (in source order: A, B, C, D, E, F, G, H, I, J):

\[
\sum_{p=1}^{111} t_{d,Pp} \cdot x_{Pp} \leq C_d
\]

Where:
- For device A: \( C_A = 3500 \)
- For device B: \( C_B = 4200 \)
- For device C: \( C_C = 4500 \)
- For device D: \( C_D = 2800 \)
- For device E: \( C_E = 3300 \)
- For device F: \( C_F = 3800 \)
- For device G: \( C_G = 4100 \)
- For device H: \( C_H = 3900 \)
- For device I: \( C_I = 4800 \)
- For device J: \( C_J = 3100 \)

Where the processing times \( t_{d,Pp} \) are as given in the device_time.csv evidence, for each device and product, in the original file order.

Variable domains:
\[
x_{Pp} \geq 0 \quad \text{for all } p = 1, \ldots, 111
\]

Summary:
- Decision variables: \( x_{Pp} \) (continuous, nonnegative), for each product \( Pp \).
- Objective: maximize total profit \( \sum_{p=1}^{111} \pi_{Pp} x_{Pp} \).
- Constraints: for each device \( d \), total processing time used by all products on device \( d \) does not exceed \( C_d \): \( \sum_{p=1}^{111} t_{d,Pp} x_{Pp} \leq C_d \).
- All coefficients and bounds are as given in the evidence above, in the original file order. No variables are integer or binary; all are continuous and nonnegative. No additional constraints are imposed.