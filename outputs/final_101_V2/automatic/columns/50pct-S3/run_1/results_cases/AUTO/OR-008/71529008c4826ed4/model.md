Let $x_i$ be the number of vehicles of type $i$ to order per day. The index $i$ corresponds to the VehicleID and VehicleType as given below.

Objective:
$$
\max \; 1200\,x_1 + 1800\,x_2 + 2500\,x_3 + 2000\,x_4 + 1500\,x_5 + 3000\,x_6 + 1000\,x_7 + 3500\,x_8 + 1600\,x_9 + 1700\,x_{10}
$$

Subject to:

1. Per-vehicle-type daily inventory limits:
\[
\begin{align*}
x_1 &\leq 100 \\
x_2 &\leq 80 \\
x_3 &\leq 120 \\
x_4 &\leq 90 \\
x_5 &\leq 50 \\
x_6 &\leq 30 \\
x_7 &\leq 110 \\
x_8 &\leq 40 \\
x_9 &\leq 60 \\
x_{10} &\leq 35 \\
\end{align*}
\]

2. Total inventory capacity constraint:
\[
x_1 + x_2 + x_3 + x_4 + x_5 + x_6 + x_7 + x_8 + x_9 + x_{10} \leq 100 + 80 + 120 + 90 + 50 + 30 + 110 + 40 + 60 + 35 = 715
\]

3. Integrality and nonnegativity:
\[
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i = 1,\ldots,10
\]

Where the mapping of $i$ to vehicle types is:

\[
\begin{align*}
x_1 &: \text{Sedans} \\
x_2 &: \text{SUVs} \\
x_3 &: \text{Electric Vehicles} \\
x_4 &: \text{Hybrid Vehicles} \\
x_5 &: \text{Trucks} \\
x_6 &: \text{Sports Cars} \\
x_7 &: \text{Compact Cars} \\
x_8 &: \text{Luxury Sedans} \\
x_9 &: \text{Vans} \\
x_{10} &: \text{Pickup Trucks} \\
\end{align*}
\]

All coefficients and limits are taken directly from the provided data, in source order.