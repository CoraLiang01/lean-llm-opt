**Mathematical Model**

Let  

- \(S\) be the set of section identifiers.  
- \(P\) be the set of product identifiers.  

Parameters  

\[
\begin{aligned}
c_i &\;=\; \text{capacity of section } i, && i\in S,\\
v_j &\;=\; \text{price (revenue per unit) of product } j, && j\in P,\\
w_j &\;=\; \text{shelf‑space requirement (weight) of product } j, && j\in P.
\end{aligned}
\]

Decision variables  

\[
x_{ij}\;=\;\text{number of units of product } j \text{ stocked in section } i,
\qquad i\in S,\; j\in P,
\]
with  
\[
x_{ij}\in \mathbb{Z}_{\ge 0}.
\]

**Objective**

\[
\max \;\sum_{i\in S}\sum_{j\in P} v_j\,x_{ij}
\]

**Constraints**

1. **Capacity limits (per section)**  

\[
\sum_{j\in P} w_j\,x_{ij}\;\le\; c_i,
\qquad \forall\, i\in S.
\]

2. **Non‑negativity and integrality**  

\[
x_{ij}\in \mathbb{Z}_{\ge 0},
\qquad \forall\, i\in S,\; j\in P.
\]

---

**Data Mapping**

| Parameter | Source Table | Column(s) | Mapping |
|-----------|--------------|-----------|---------|
| \(c_i\) | `file_0_view_0` | `SectionID`, `Capacity` | \(c_{\text{SectionID}} = \text{Capacity}\) |
| \(v_j\) | `file_1_view_0` | `ProductName`, `Value` | \(v_{\text{ProductName}} = \text{Value}\) |
| \(w_j\) | `file_1_view_0` | `ProductName`, `Weight` | \(w_{\text{ProductName}} = \text{Weight}\) |

*Index sets*  
- \(S = \{\text{SectionID values from `file_0_view_0`}\}\)  
- \(P = \{\text{ProductName values from `file_1_view_0`}\}\)

All parameters are taken directly from the corresponding CSV files; no numerical values are enumerated in the model.