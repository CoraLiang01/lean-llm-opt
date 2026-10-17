**Mathematical Model**

\[
\begin{aligned}
\text{minimize}\quad & Z \;=\; \sum_{s\in S}\;\sum_{d\in D}\; \text{cost}_{s,d}\;x_{s,d} \\[4pt]
\text{subject to}\quad
&\sum_{s\in S} x_{s,d}\;=\;\text{demand}_{d}\qquad &&\forall d\in D \quad\text{(demand)}\\[4pt]
&\sum_{d\in D} x_{s,d}\;\le\;\text{supply}_{s}\qquad &&\forall s\in S \quad\text{(supply)}\\[4pt]
&x_{s,d}\;\in\;\mathbb{Z}_{\ge 0}\qquad &&\forall s\in S,\;d\in D
\end{aligned}
\]

---

### Data Mapping

| Symbol | Source Table | Column(s) | Notes |
|--------|--------------|-----------|-------|
| \(S\) | `file_1_view_0` | `Unnamed: 0` | Warehouse identifiers (S1–S10) |
| \(D\) | `file_0_view_0` | `customer` | Customer identifiers (C1–C10) |
| \(\text{supply}_{s}\) | `file_1_view_0` | `supply_capacity` | Supply capacity of warehouse \(s\) |
| \(\text{demand}_{d}\) | `file_0_view_0` | `demand` | Demand of customer \(d\) |
| \(\text{cost}_{s,d}\) | `file_2_view_0` | `C1`–`C10` | Transportation cost from warehouse \(s\) to customer \(d\) |

All parameters are taken directly from the corresponding CSV tables; no additional data are introduced. The model uses integer decision variables \(x_{s,d}\) to represent whole units shipped.