##### Decision Variables

$x_i \in \mathbb{Z}_{\geq 0}$: Number of vehicles of type $i$ to order per day, for each vehicle type $i$ in the set
$$
I = \{\text{Sedans},\ \text{SUVs},\ \text{Electric Vehicles},\ \text{Hybrid Vehicles},\ \text{Trucks},\ \text{Sports Cars},\ \text{Compact Cars},\ \text{Luxury Sedans},\ \text{Vans},\ \text{Pickup Trucks}\}
$$

##### Parameters

- Benefit coefficients $b_i$ (from products.csv):

  - $b_{\text{Sedans}} = 1200$
  - $b_{\text{SUVs}} = 1800$
  - $b_{\text{Electric Vehicles}} = 2500$
  - $b_{\text{Hybrid Vehicles}} = 2000$
  - $b_{\text{Trucks}} = 1500$
  - $b_{\text{Sports Cars}} = 3000$
  - $b_{\text{Compact Cars}} = 1000$
  - $b_{\text{Luxury Sedans}} = 3500$
  - $b_{\text{Vans}} = 1600$
  - $b_{\text{Pickup Trucks}} = 1700$

- Daily inventory limits $u_i$ (from capacity.csv):

  - $u_{\text{Sedans}} = 100$
  - $u_{\text{SUVs}} = 80$
  - $u_{\text{Electric Vehicles}} = 120$
  - $u_{\text{Hybrid Vehicles}} = 90$
  - $u_{\text{Trucks}} = 50$
  - $u_{\text{Sports Cars}} = 30$
  - $u_{\text{Compact Cars}} = 110$
  - $u_{\text{Luxury Sedans}} = 40$
  - $u_{\text{Vans}} = 60$
  - $u_{\text{Pickup Trucks}} = 35$

- Total inventory capacity: Not specified; only per-vehicle-type limits are enforced.

##### Objective Function

$$
\max \sum_{i \in I} b_i x_i
$$

##### Constraints

1. Daily inventory limit for each vehicle type:
   $$
   0 \leq x_i \leq u_i, \quad \forall i \in I
   $$
2. Integrality:
   $$
   x_i \in \mathbb{Z}_{\geq 0}, \quad \forall i \in I
   $$

##### Complete Model

$$
\begin{align*}
\max\quad & 1200\,x_{\text{Sedans}} + 1800\,x_{\text{SUVs}} + 2500\,x_{\text{Electric Vehicles}} + 2000\,x_{\text{Hybrid Vehicles}} \\
& + 1500\,x_{\text{Trucks}} + 3000\,x_{\text{Sports Cars}} + 1000\,x_{\text{Compact Cars}} + 3500\,x_{\text{Luxury Sedans}} \\
& + 1600\,x_{\text{Vans}} + 1700\,x_{\text{Pickup Trucks}} \\
\text{s.t.}\quad
& 0 \leq x_{\text{Sedans}} \leq 100 \\
& 0 \leq x_{\text{SUVs}} \leq 80 \\
& 0 \leq x_{\text{Electric Vehicles}} \leq 120 \\
& 0 \leq x_{\text{Hybrid Vehicles}} \leq 90 \\
& 0 \leq x_{\text{Trucks}} \leq 50 \\
& 0 \leq x_{\text{Sports Cars}} \leq 30 \\
& 0 \leq x_{\text{Compact Cars}} \leq 110 \\
& 0 \leq x_{\text{Luxury Sedans}} \leq 40 \\
& 0 \leq x_{\text{Vans}} \leq 60 \\
& 0 \leq x_{\text{Pickup Trucks}} \leq 35 \\
& x_i \in \mathbb{Z}_{\geq 0},\quad \forall i \in I
\end{align*}
$$

All parameters and constraints are derived directly from the provided CSV data. If a total inventory capacity constraint is later specified, add:
$$
\sum_{i \in I} x_i \leq \text{Total Inventory Capacity}
$$
but this is not present in the current data.