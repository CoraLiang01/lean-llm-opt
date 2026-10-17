**Mathematical Model**

Let  

* \(I=\{i\mid i\text{ is a row id in } \texttt{file\_1\_view\_0}\}\) – candidate locations  
* \(J=\{j\mid j\text{ is a column id in } \texttt{file\_0\_view\_0}\}\) – residential areas  

Parameters  

* \(d_j\) – demand of area \(j\) (from \(\texttt{file\_0\_view\_0}\))  
* \(c_{ij}\) – distance from location \(i\) to area \(j\) (from \(\texttt{file\_1\_view\_0}\))  
* \(P\) – number of facilities to open (from \(\texttt{file\_2\_view\_0}\))  

Decision variables  

* \(y_i \in \{0,1\}\) – 1 if location \(i\) is opened  
* \(x_{ij} \in \{0,1\}\) – 1 if area \(j\) is assigned to location \(i\)

\[
\begin{aligned}
\min\;& \sum_{i\in I}\sum_{j\in J} d_j\,c_{ij}\,x_{ij} \\[4pt]
\text{s.t. } & \sum_{i\in I} x_{ij}=1, &&\forall j\in J \quad\text{(exact‑one assignment)}\\[4pt]
& \sum_{i\in I} y_i = P, &&\text{(exactly \(P\) facilities)}\\[4pt]
& x_{ij}\le y_i, &&\forall i\in I,\; j\in J \quad\text{(assignment only to opened sites)}\\[4pt]
& x_{ij}\in\{0,1\}, &&\forall i\in I,\; j\in J\\
& y_i\in\{0,1\}, &&\forall i\in I
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Source Column(s) | Notes |
|-----------|--------------|------------------|-------|
| \(d_j\) | `file_0_view_0` | `Demand` | Row id = `Area` |
| \(c_{ij}\) | `file_1_view_0` | Matrix cells | Row id = `Location`, column id = each `Area` |
| \(P\) | `file_2_view_0` | `Value` | Row where `Parameter` = `NumberOfFacilitiesToOpen` |

Index sets are derived from the row and column identifiers of the corresponding tables. No numerical values are enumerated in the model; all coefficients are linked to the source data via the mapping above.