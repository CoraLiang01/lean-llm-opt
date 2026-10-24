**Mathematical Model**

Let  

- \(I\) = set of storage areas indexed by **StorageID** (from *file_0_view_0*).  
- \(J\) = set of air‑conditioner types indexed by **ProductName** (from *file_1_view_0*).  

Parameters  

- \(C_i\) – capacity of storage area \(i\) (units of weight).  
- \(v_j\) – value of product \(j\).  
- \(w_j\) – weight (size) of product \(j\).  

Decision variables  

- \(x_{ij}\) – number of units of product \(j\) placed in storage area \(i\).  

\[
\begin{aligned}
\max\;& \sum_{i\in I}\sum_{j\in J} v_j\,x_{ij} \\
\text{s.t. } & \sum_{j\in J} w_j\,x_{ij} \;\le\; C_i , && \forall i\in I \\
& x_{ij}\;\in\;\mathbb{Z}_{\ge 0}, && \forall i\in I,\; j\in J .
\end{aligned}
\]

---

**Data Mapping**

| Symbol | Source Table | Column |
|--------|---------------|--------|
| \(I\) | `file_0_view_0` | `StorageID` |
| \(J\) | `file_1_view_0` | `ProductName` |
| \(C_i\) | `file_0_view_0` | `Capacity` |
| \(v_j\) | `file_1_view_0` | `Value` |
| \(w_j\) | `file_1_view_0` | `Weight` |

All other model components are defined symbolically; no numerical values are hard‑coded.