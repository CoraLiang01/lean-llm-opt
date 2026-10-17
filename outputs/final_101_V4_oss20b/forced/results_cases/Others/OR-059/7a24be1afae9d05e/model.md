**Mathematical Model**

Let  

- \(I=\{S1,\dots ,S8\}\) suppliers  
- \(J=\{C1,\dots ,C9\}\) dealerships  

Parameters  

- \(f_i\) fixed opening cost of supplier \(i\) (\$)  
- \(t_{ij}\) transportation cost per vehicle from supplier \(i\) to dealership \(j\) (\$)  
- \(d_j\) demand of dealership \(j\) (vehicles)  

Decision variables  

- \(y_i\in\{0,1\}\) 1 if supplier \(i\) is opened, 0 otherwise  
- \(x_{ij}\ge 0\) number of vehicles supplied by \(i\) to \(j\)

\[
\begin{aligned}
\min\;&\sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} t_{ij}\,x_{ij} \\[4pt]
\text{s.t. } &\sum_{i\in I} x_{ij} \;\ge\; d_j \qquad \forall j\in J \quad\text{(meet demand)}\\[4pt]
&x_{ij}\;\le\; d_j\,y_i \qquad \forall i\in I,\; j\in J \quad\text{(linking)}\\[4pt]
&y_i\in\{0,1\}\qquad \forall i\in I\\
&x_{ij}\ge 0 \qquad \forall i\in I,\; j\in J
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Source Table | Source Column | Mapping Rule |
|-----------|--------------|---------------|--------------|
| \(f_i\) | `file_1_view_0` | `fixed_costs` | For each row, `Unnamed: 0` → supplier id \(i\); value → \(f_i\) |
| \(t_{ij}\) | `file_2_view_0` | `C1`–`C9` | For each row, `Unnamed: 0` → supplier id \(i\); each column `Cj` → dealership id \(j\); value → \(t_{ij}\) |
| \(d_j\) | `file_0_view_0` | `demand` | For each row, `customer` → dealership id \(j\); value → \(d_j\) |

All indices \(i\) and \(j\) are taken from the complete sets of suppliers and dealerships present in the respective tables. No additional filtering or aggregation is applied.