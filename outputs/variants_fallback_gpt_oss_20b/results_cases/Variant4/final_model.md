**Symbolic Model**

Let  

- \(I\) be the set of candidate centers.  
- \(J\) be the set of districts.  
- \(f_i\) be the opening cost of center \(i\).  
- \(a_{ij}\in\{0,1\}\) be 1 if center \(i\) can cover district \(j\), 0 otherwise.  
- \(y_i\in\{0,1\}\) be 1 if center \(i\) is opened, 0 otherwise.  

\[
\begin{aligned}
\min\;&\sum_{i\in I} f_i\,y_i \\[4pt]
\text{s.t.}\;&\sum_{i\in I} a_{ij}\,y_i \;\ge\; 1, &&\forall j\in J \\[4pt]
&y_i\in\{0,1\}, &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Table ID | Column(s) | Mapping Rule |
|-----------|--------|----------|-----------|--------------|
| \(I\) | Centers | `file_0_view_0` | `Center` | Set of all distinct `Center` values |
| \(J\) | Districts | `file_1_view_0` | `District` | Set of all distinct `District` values |
| \(f_i\) | Opening cost | `file_0_view_0` | `OpeningCost` | \(f_i =\) value of `OpeningCost` for center \(i\) |
| \(a_{ij}\) | Coverage | `file_0_view_0` | `CoveredDistricts` | \(a_{ij}=1\) if district \(j\) appears in the semicolon‑separated list `CoveredDistricts` for center \(i\); otherwise \(a_{ij}=0\) |
| \(y_i\) | Decision variable | – | – | Binary variable defined for each \(i\in I\) |

All other data (e.g., district list) are used only to define the index set \(J\). No numerical values are enumerated in the model.