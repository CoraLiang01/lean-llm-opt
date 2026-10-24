**Symbolic Model**

Let  

- \(I=\{i\}\) be the set of candidate centers (from *service_centers.csv*).  
- \(J=\{j\}\) be the set of districts that must be covered (from *districts.csv*).  

Parameters  

- \(f_i\) – opening cost of center \(i\).  
- \(a_{ij}\in\{0,1\}\) – equals 1 if center \(i\) can cover district \(j\), 0 otherwise.  

Decision variables  

- \(y_i\in\{0,1\}\) – 1 if center \(i\) is opened, 0 otherwise.  

\[
\begin{aligned}
\min\;&\sum_{i\in I} f_i\,y_i \\[4pt]
\text{s.t. } &\sum_{i\in I} a_{ij}\,y_i \;\ge\; 1 , &&\forall j\in J \quad\text{(coverage)}\\[4pt]
& y_i \in \{0,1\}, &&\forall i\in I .
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source | Mapping Rule |
|-----------|--------|--------------|
| \(I\) | `file_0_view_0` – column **Center** | Each distinct value of **Center** forms an index \(i\). |
| \(J\) | `file_1_view_0` – column **District** | Each distinct value of **District** forms an index \(j\). |
| \(f_i\) | `file_0_view_0` – column **OpeningCost** | \(f_i =\) numeric value in **OpeningCost** for center \(i\). |
| \(a_{ij}\) | `file_0_view_0` – column **CoveredDistricts** | For each center \(i\), split the semicolon‑separated string in **CoveredDistricts** into district identifiers. Set \(a_{ij}=1\) if district \(j\) appears in that list; otherwise \(a_{ij}=0\). |

All other data (e.g., district list) are used only to define the index set \(J\). No additional constraints or capacities are required.