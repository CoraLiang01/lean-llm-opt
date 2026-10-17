**Symbolic Linear‑Programming Model**

Let  

* \(T=\{0,1,\dots ,47\}\) : index of possible 30‑minute start times for a shift  
* \(P=\{0,1,\dots ,47\}\) : index of 30‑minute periods in the 24‑hour day  

Parameters  

* \(R_p\) : minimum number of waitstaff required in period \(p\)  
  *\(R_p\) is taken from table **file_0_view_0**, column **Requirement**, row **source_row = p***  

* \(S_{t,p}\) : 1 if a shift that starts at time \(t\) covers period \(p\), 0 otherwise  
  *\(S_{t,p}=1\) if \((p-t)\bmod 48 < 16\) (an 8‑hour shift = 16 half‑hour slots)  
  *\(S_{t,p}=0\) otherwise  

Decision variables  

* \(y_t \in \{0,1\}\) : 1 if a waitstaff starts a shift at time \(t\), 0 otherwise  

Objective  

\[
\min \sum_{t\in T} y_t
\]

Coverage constraints (for every period \(p\))  

\[
\sum_{t\in T} S_{t,p}\,y_t \;\ge\; R_p \qquad \forall\,p\in P
\]

Variable domain  

\[
y_t \in \{0,1\}\qquad \forall\,t\in T
\]

---

**Data Mapping**

| Parameter | Source | Mapping |
|-----------|--------|---------|
| \(R_p\) | **file_0_view_0** | `Requirement` column, row `source_row = p` |
| \(S_{t,p}\) | Defined by rule | \(S_{t,p}=1\) if \((p-t)\bmod 48 < 16\), else 0 |

No numerical values are enumerated; the model references the data table and the shift‑coverage rule directly.