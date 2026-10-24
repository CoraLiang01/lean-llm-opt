**Mathematical Model**

Sets  
\[
S=\{s\mid s\in \text{file\_1\_view\_0.region}\},\qquad
C=\{c\mid c\in \text{file\_0\_view\_0.customer}\}
\]

Parameters  
\[
\begin{aligned}
\text{demand}[c] &:= \text{file\_0\_view\_0.demand}\quad(c\in C)\\
\text{supply}[s] &:= \text{file\_1\_view\_0.supply\_capacity}\quad(s\in S)\\
\text{cost}[s,c] &:= \text{file\_2\_view\_0}[s,c]\quad(s\in S,\;c\in C)
\end{aligned}
\]

Variables  
\[
x_{s,c}\in \mathbb{Z}_{\ge 0}\quad(s\in S,\;c\in C)
\]

Objective  
\[
\min \;\sum_{s\in S}\sum_{c\in C}\text{cost}[s,c]\;x_{s,c}
\]

Constraints  

Demand satisfaction  
\[
\sum_{s\in S}x_{s,c}= \text{demand}[c]\qquad \forall\,c\in C
\]

Supply capacity  
\[
\sum_{c\in C}x_{s,c}\le \text{supply}[s]\qquad \forall\,s\in S
\]

Non‑negativity (implicit in variable domain).  

---

**Data Mapping**

| Parameter | Table ID | Column(s) |
|-----------|----------|-----------|
| `demand[c]` | `file_0_view_0` | `demand` |
| `supply[s]` | `file_1_view_0` | `supply_capacity` |
| `cost[s,c]` | `file_2_view_0` | Row axis: `Unnamed: 0` (warehouse ID), Column axis: `D1`–`D5` (store IDs) |

| Set | Table ID | Column |
|-----|----------|--------|
| `S` | `file_1_view_0` | `region` |
| `C` | `file_0_view_0` | `customer` |

All indices, parameters, and variables are defined exactly as above; no numerical values are enumerated.