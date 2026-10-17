**Mathematical Model**

Let  

- \(I=\{1,\dots ,10\}\) be the set of platforms.  
- \(J=\{1,\dots ,15\}\) be the set of product genres.  

Parameters  

- \(c_i\) – memory capacity of platform \(i\).  
- \(v_j\) – value of one unit of product \(j\).  
- \(w_j\) – memory weight of one unit of product \(j\).  

Decision variables  

- \(x_{ij}\in \mathbb{Z}_{\ge 0}\) – number of units of product \(j\) listed on platform \(i\).

\[
\begin{aligned}
\max\;& \sum_{i\in I}\sum_{j\in J} v_j\,x_{ij} \\[4pt]
\text{s.t. } & \sum_{j\in J} w_j\,x_{ij}\;\le\; c_i , && \forall i\in I \\[4pt]
& x_{ij}\;\in\;\mathbb{Z}_{\ge 0}, && \forall i\in I,\; j\in J
\end{aligned}
\]

---

**Data Mapping Rules**

| Parameter | Source Table | Key Column | Value Column | Mapping |
|-----------|--------------|------------|--------------|---------|
| \(c_i\) | `file_0_view_0` | `PlatformId` | `Capacity` | \(c_i = \text{Capacity}\) where `PlatformId` = \(i\) |
| \(v_j\) | `file_1_view_0` | `ProductName` | `Value` | \(v_j = \text{Value}\) where `ProductName` = \(j\) |
| \(w_j\) | `file_1_view_0` | `ProductName` | `Weight` | \(w_j = \text{Weight}\) where `ProductName` = \(j\) |

*Indices \(i\) and \(j\) are taken directly from the distinct values of `PlatformId` and `ProductName` in the respective tables.*