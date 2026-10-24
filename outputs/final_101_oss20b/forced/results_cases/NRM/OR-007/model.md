**Mathematical Model**

Let  

- \(W=\{S1,S2,S3,S4,S5\}\) (warehouses)  
- \(S=\{D1,D2,D3,D4,D5\}\) (stores)

Parameters  

\[
\begin{aligned}
&\text{demand}_s &&\forall s\in S \\
&\text{supply}_w &&\forall w\in W \\
&\text{cost}_{w,s} &&\forall w\in W,\; s\in S
\end{aligned}
\]

Decision variables  

\[
x_{w,s}\in \mathbb{Z}_{\ge 0}\qquad \forall w\in W,\; s\in S
\]

Objective  

\[
\min \;\sum_{w\in W}\sum_{s\in S}\text{cost}_{w,s}\,x_{w,s}
\]

Constraints  

1. **Demand satisfaction**  
\[
\sum_{w\in W}x_{w,s}= \text{demand}_s \qquad \forall s\in S
\]

2. **Supply capacity**  
\[
\sum_{s\in S}x_{w,s}\le \text{supply}_w \qquad \forall w\in W
\]

3. **Non‑negativity & integrality**  
\[
x_{w,s}\in \mathbb{Z}_{\ge 0}\qquad \forall w\in W,\; s\in S
\]

---

**Data Mapping**

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(\text{demand}_s\) | `file_0_view_0` | `demand` (row identified by `customer` = s) |
| \(\text{supply}_w\) | `file_1_view_0` | `supply_capacity` (row identified by `region` = w) |
| \(\text{cost}_{w,s}\) | `file_2_view_0` | `D1`–`D5` (columns) with row identified by `Unnamed: 0` = w |

All indices \(w\) and \(s\) are taken from the row and column identifiers of the respective tables. No additional data or assumptions are introduced.