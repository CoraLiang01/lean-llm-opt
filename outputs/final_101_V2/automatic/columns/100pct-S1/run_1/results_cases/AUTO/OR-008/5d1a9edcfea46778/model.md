Let $x_i$ be the number of vehicles of type $i$ to order per day. The index $i$ corresponds to the vehicle types as listed in the data.

Objective:
$$
\max \; 1200\,x_1 + 1800\,x_2 + 2500\,x_3 + 2000\,x_4 + 1500\,x_5 + 3000\,x_6 + 1000\,x_7 + 3500\,x_8 + 1600\,x_9 + 1700\,x_{10}
$$

Subject to:

Per-vehicle-type daily inventory limits:
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

Total inventory capacity constraint:
\[
x_1 + x_2 + x_3 + x_4 + x_5 + x_6 + x_7 + x_8 + x_9 + x_{10} \leq C
\]
where $C$ is the total inventory capacity per day (to be specified if provided; if not, this constraint is omitted).

Nonnegativity and integrality:
\[
x_i \in \mathbb{Z}_{\geq 0} \quad \forall i = 1,\ldots,10
\]

Mapping of indices to vehicle types:
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

All coefficients and limits are taken directly from the provided data, preserving source order and identifiers.