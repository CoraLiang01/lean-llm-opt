**Mathematical Model**

Let  

* \(P\) – set of platforms (indices from *file_0_view_0*).  
* \(G\) – set of products (indices from *file_1_view_0*).  

Parameters  

\[
\begin{aligned}
c_p &\;=\;\text{Capacity of platform }p \quad (p\in P)\\
w_g &\;=\;\text{Weight (memory) of product }g \quad (g\in G)\\
v_g &\;=\;\text{Value of product }g \quad (g\in G)
\end{aligned}
\]

Decision variables  

\[
x_{p,g}\;\in\;\mathbb{Z}_{\ge 0}\qquad (p\in P,\;g\in G)
\]

---

**Objective**

\[
\max \;\sum_{p\in P}\sum_{g\in G} v_g\,x_{p,g}
\]

---

**Constraints**

1. **Capacity limits** – for every platform \(p\in P\)

\[
\sum_{g\in G} w_g\,x_{p,g}\;\le\; c_p
\]

2. **Non‑negativity & integrality** – already encoded in the variable definition.

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(c_p\)   | `file_0_view_0` | `Capacity` |
| \(w_g\)   | `file_1_view_0` | `Weight` |
| \(v_g\)   | `file_1_view_0` | `Value` |

*Index sets*  
- \(P\) is the set of all `PlatformID` values in `file_0_view_0`.  
- \(G\) is the set of all `ProductName` values in `file_1_view_0`.

No additional constraints or parameters are required.