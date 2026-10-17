Let $x_i$ denote the production quantity (in units) of product $i$ ($i \in \{\text{P1}, \text{P2}, \ldots, \text{P111}\}$).

Parameters:
- $p_i$: Unit profit of product $i$ (from unit_product_profits.csv)
- $a_{ji}$: Processing time required by product $i$ on device $j$ (from device_time.csv, $j \in \{\text{A}, \text{B}, \ldots, \text{J}\}$)
- $c_j$: Monthly operating capacity of device $j$ (from monthly_device_capacity.csv)

Decision variables:
- $x_i \geq 0$, continuous, for all $i \in \{\text{P1}, \ldots, \text{P111}\}$

Objective:
\[
\max \sum_{i=\text{P1}}^{\text{P111}} p_i x_i
\]

Subject to, for each device $j \in \{\text{A}, \ldots, \text{J}\}$:
\[
\sum_{i=\text{P1}}^{\text{P111}} a_{ji} x_i \leq c_j
\]

\[
x_i \geq 0 \quad \forall i \in \{\text{P1}, \ldots, \text{P111}\}
\]

Where:

- $p_i$ is given by the "Unit_Profit" column in unit_product_profits.csv for each product $i$.
- $a_{ji}$ is given by the entry in device_time.csv at row "Device" $j$ and column $i$.
- $c_j$ is given by the "Monthly_Capacity" column in monthly_device_capacity.csv for each device $j$.

Explicitly, the model is:

\[
\max \left(
28.55\,x_{\text{P1}} + 12.78\,x_{\text{P2}} + 45.21\,x_{\text{P3}} + \cdots + 9.99\,x_{\text{P111}}
\right)
\]

Subject to, for each device $j$:

For device A:
\[
8.1\,x_{\text{P1}} + 2.5\,x_{\text{P2}} + 10.2\,x_{\text{P3}} + \cdots + 8.5\,x_{\text{P111}} \leq 3500
\]

For device B:
\[
10.5\,x_{\text{P1}} + 5.2\,x_{\text{P2}} + 8.3\,x_{\text{P3}} + \cdots + 8.7\,x_{\text{P111}} \leq 4200
\]

For device C:
\[
2.1\,x_{\text{P1}} + 13.4\,x_{\text{P2}} + 10.3\,x_{\text{P3}} + \cdots + 2.9\,x_{\text{P111}} \leq 4500
\]

For device D:
\[
5.8\,x_{\text{P1}} + 1.2\,x_{\text{P2}} + 11.7\,x_{\text{P3}} + \cdots + 8.3\,x_{\text{P111}} \leq 2800
\]

For device E:
\[
9.3\,x_{\text{P1}} + 4.1\,x_{\text{P2}} + 12.8\,x_{\text{P3}} + \cdots + 6.7\,x_{\text{P111}} \leq 3300
\]

For device F:
\[
3.8\,x_{\text{P1}} + 14.2\,x_{\text{P2}} + 9.7\,x_{\text{P3}} + \cdots + 8.1\,x_{\text{P111}} \leq 3800
\]

For device G:
\[
7.2\,x_{\text{P1}} + 2.8\,x_{\text{P2}} + 13.1\,x_{\text{P3}} + \cdots + 5.9\,x_{\text{P111}} \leq 4100
\]

For device H:
\[
11.7\,x_{\text{P1}} + 6.3\,x_{\text{P2}} + 1.9\,x_{\text{P3}} + \cdots + 1.4\,x_{\text{P111}} \leq 3900
\]

For device I:
\[
1.1\,x_{\text{P1}} + 11.3\,x_{\text{P2}} + 6.9\,x_{\text{P3}} + \cdots + 7.3\,x_{\text{P111}} \leq 4800
\]

For device J:
\[
4.6\,x_{\text{P1}} + 0.2\,x_{\text{P2}} + 10.4\,x_{\text{P3}} + \cdots + 12.4\,x_{\text{P111}} \leq 3100
\]

And for all $i$:
\[
x_i \geq 0
\]

All coefficients and identifiers are as given in the supplied CSV files and preserved in source order. The variables $x_i$ are nonnegative and continuous. The objective is to maximize total monthly profit, subject to device processing time capacity constraints.