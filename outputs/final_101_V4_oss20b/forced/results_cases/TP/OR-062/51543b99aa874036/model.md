**Mathematical Model**

Let  

* \(I=\{i\}\) – set of suppliers (MOUNT AYR, WAUKEE, WAVERLY, PELLA, DES MOINES)  
* \(J=\{j\}\) – set of licensed stores (CLARINDA, FORT MADISON, SIOUX CITY, TOLEDO, BANCROFT)  

Parameters (all real, non‑negative)

* \(f_i\) – fixed opening cost of supplier \(i\)  
* \(d_j\) – demand of store \(j\)  
* \(c_{ij}\) – unit transportation cost from supplier \(i\) to store \(j\)  
* \(M=\sum_{j\in J} d_j\) – a sufficiently large constant (here \(M=11835\))

Decision variables  

* \(y_i\in\{0,1\}\) – 1 if supplier \(i\) is opened, 0 otherwise  
* \(x_{ij}\ge 0\) – quantity shipped from supplier \(i\) to store \(j\)

\[
\begin{aligned}
\min\;&\sum_{i\in I} f_i\,y_i \;+\; \sum_{i\in I}\sum_{j\in J} c_{ij}\,x_{ij} \\[4pt]
\text{s.t. } &\sum_{i\in I} x_{ij} \;\ge\; d_j , &&\forall j\in J \quad\text{(demand)}\\[4pt]
&x_{ij}\;\le\; M\,y_i , &&\forall i\in I,\; j\in J \quad\text{(linking)}\\[4pt]
&y_i\in\{0,1\}, &&\forall i\in I\\
&x_{ij}\ge 0 , &&\forall i\in I,\; j\in J
\end{aligned}
\]

---

**Data Mapping Rules**

| Table ID | Source Columns | Parameter Mapping |
|----------|----------------|-------------------|
| `file_0_view_0` | `Customer` → \(j\) (store index) <br> `demand` → \(d_j\) | `demand` parameter |
| `file_1_view_0` | `Unnamed: 0` → \(i\) (supplier index) <br> `fixed_costs` → \(f_i\) | `fixed_cost` parameter |
| `file_2_view_0` | `Unnamed: 0` → \(i\) (supplier index) <br> `CLARINDA`, `FORT MADISON`, `SIOUX CITY`, `TOLEDO`, `BANCROFT` → \(j\) (store indices) <br> cell value → \(c_{ij}\) | `transportation_cost` matrix |

*All indices and values are taken verbatim from the corresponding CSV files; no aggregation or transformation is performed beyond the big‑\(M\) linking constraint.*