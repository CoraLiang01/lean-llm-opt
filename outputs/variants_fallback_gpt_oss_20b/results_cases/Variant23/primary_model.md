**Mathematical Model**

\[
\begin{aligned}
\min\;&\sum_{d\in D}\sum_{m\in M}\Bigl(c_{dm}\,x_{dm}+f_{dm}\,y_{dm}\Bigr)\\[4pt]
\text{s.t. }&\sum_{d\in D}x_{dm}=d_m \qquad &&\forall m\in M \quad\text{(market demand)}\\[4pt]
&\sum_{m\in M}x_{dm}\le s_d \qquad &&\forall d\in D \quad\text{(depot supply)}\\[4pt]
&x_{dm}\le M_{dm}\,y_{dm} \qquad &&\forall d\in D,\;m\in M \quad\text{(linking)}\\[4pt]
&x_{dm}\ge 0 \qquad &&\forall d\in D,\;m\in M\\[4pt]
&y_{dm}\in\{0,1\} \qquad &&\forall d\in D,\;m\in M
\end{aligned}
\]

where  

\[
M_{dm}= \min\{\,s_d,\;d_m\,\}.
\]

---

### Data Mapping

| Index | Source | Table ID | Column(s) | Notes |
|-------|--------|----------|-----------|-------|
| \(D\) | Depots | `file_0_view_0` | `Depot` | Set of all depots |
| \(M\) | Markets | `file_1_view_0` | `Market` | Set of all markets |
| \(s_d\) | Depot supply | `file_0_view_0` | `SupplyCapacity` | Capacity of depot \(d\) |
| \(d_m\) | Market demand | `file_1_view_0` | `Demand` | Demand of market \(m\) |
| \(c_{dm}\) | Variable cost | `file_2_view_0` | `M1`–`M4` | Per‑unit shipping cost from depot \(d\) to market \(m\) |
| \(f_{dm}\) | Fixed cost | `file_3_view_0` | `M1`–`M4` | Activation cost of route \(d\!\to\!m\) |
| \(M_{dm}\) | Big‑\(M\) | derived | \(\min(s_d,d_m)\) | Upper bound linking \(x_{dm}\) and \(y_{dm}\) |

All parameters are taken directly from the corresponding CSV files; no additional data are invented. The model uses the full set of depots and markets present in the source tables, and the linking constraints enforce that a shipment on a route can occur only if that route is activated.