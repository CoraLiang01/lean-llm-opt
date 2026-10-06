Let $x_i$ be the number of vehicles of type $i$ to order per day. The index $i$ corresponds to the VehicleType as given in the data.

Objective:
$$
\max \; 1200\,x_1 + 1800\,x_2 + 2500\,x_3 + 2000\,x_4 + 1500\,x_5 + 3000\,x_6 + 1000\,x_7 + 3500\,x_8 + 1600\,x_9 + 1700\,x_{10}
$$

Subject to:

1. Per-vehicle-type daily inventory limits:
\[
\begin{align*}
x_1 &\leq 100 \quad &\text{(Sedans)} \\
x_2 &\leq 80 \quad &\text{(SUVs)} \\
x_3 &\leq 120 \quad &\text{(Electric Vehicles)} \\
x_4 &\leq 90 \quad &\text{(Hybrid Vehicles)} \\
x_5 &\leq 50 \quad &\text{(Trucks)} \\
x_6 &\leq 30 \quad &\text{(Sports Cars)} \\
x_7 &\leq 110 \quad &\text{(Compact Cars)} \\
x_8 &\leq 40 \quad &\text{(Luxury Sedans)} \\
x_9 &\leq 60 \quad &\text{(Vans)} \\
x_{10} &\leq 35 \quad &\text{(Pickup Trucks)} \\
\end{align*}
\]

2. Total inventory capacity constraint:
\[
x_1 + x_2 + x_3 + x_4 + x_5 + x_6 + x_7 + x_8 + x_9 + x_{10} \leq 100 + 80 + 120 + 90 + 50 + 30 + 110 + 40 + 60 + 35 = 715
\]

3. Nonnegativity and integrality:
\[
x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i = 1,\ldots,10
\]

Where the mapping of $i$ to VehicleType is:

\[
\begin{align*}
1 &:\ \text{Sedans} \\
2 &:\ \text{SUVs} \\
3 &:\ \text{Electric Vehicles} \\
4 &:\ \text{Hybrid Vehicles} \\
5 &:\ \text{Trucks} \\
6 &:\ \text{Sports Cars} \\
7 &:\ \text{Compact Cars} \\
8 &:\ \text{Luxury Sedans} \\
9 &:\ \text{Vans} \\
10 &:\ \text{Pickup Trucks} \\
\end{align*}
\]

All coefficients and limits are taken directly from the retrieved data, in source order.