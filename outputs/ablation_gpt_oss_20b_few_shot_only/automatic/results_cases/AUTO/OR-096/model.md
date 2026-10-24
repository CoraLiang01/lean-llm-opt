**Mathematical Model**

Let  

* \(S=\{I,II\}\) – schools  
* \(N=\{N01,\dots ,N31\}\) – neighborhoods  

Parameters  

* \(C_s\) – capacity of school \(s\)  (from *school_capacity.csv*)  
* \(W_n\) – white‑student population of neighborhood \(n\)  (from *neighborhoods_population.csv*)  
* \(NW_n\) – non‑white‑student population of neighborhood \(n\)  (from *neighborhoods_population.csv*)  
* \(d_{s,n}\) – distance (miles) from school \(s\) to neighborhood \(n\)  (from *distance.csv*)  
* \(r=0.60\) – district white‑student ratio  
* \(\tau=0.10\) – allowed deviation  

Decision variables  

\[
x_{s,n}^{W}\;\ge 0 \quad\text{(white pupils from }n\text{ assigned to }s)
\]
\[
x_{s,n}^{NW}\;\ge 0 \quad\text{(non‑white pupils from }n\text{ assigned to }s)
\]

Objective  

\[
\min \;\sum_{s\in S}\sum_{n\in N} d_{s,n}\,\bigl(x_{s,n}^{W}+x_{s,n}^{NW}\bigr)
\]

Constraints  

1. **School capacity**  
\[
\sum_{n\in N}\bigl(x_{s,n}^{W}+x_{s,n}^{NW}\bigr)\;\le\; C_s
\qquad \forall s\in S
\]

2. **Neighborhood population balance**  
\[
\sum_{s\in S}x_{s,n}^{W}=W_n
\qquad \forall n\in N
\]
\[
\sum_{s\in S}x_{s,n}^{NW}=NW_n
\qquad \forall n\in N
\]

3. **Racial‑balance at each school**  
Let \(T_s=\sum_{n}(x_{s,n}^{W}+x_{s,n}^{NW})\).  
\[
r-\tau \;\le\; \frac{\sum_{n}x_{s,n}^{W}}{T_s}\;\le\; r+\tau
\qquad \forall s\in S
\]
Equivalently (to keep linearity)  
\[
(r-\tau)\,T_s \;\le\; \sum_{n}x_{s,n}^{W}
\;\le\; (r+\tau)\,T_s
\qquad \forall s\in S
\]

All variables are non‑negative continuous.

---

**Data Mapping**

| Parameter | Source File | Column(s) | Mapping |
|-----------|-------------|-----------|---------|
| \(C_s\) | `Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/school_capacity.csv` | `School`, `Capacity` | `C_I = 2028`, `C_II = 1560` |
| \(W_n\) | `Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/neighborhoods_population.csv` | `Neighborhood`, `Population_White` | `W_N01 = 78`, …, `W_N31 = 74` |
| \(NW_n\) | same file | `Neighborhood`, `Population_NonWhite` | `NW_N01 = 22`, …, `NW_N31 = 46` |
| \(d_{s,n}\) | `Test_Dataset/Large-scale-or/Mixture_testing/Mixture15/distance.csv` | Row `School`, columns `N01`–`N31` | `d_I,N01 = 1.25`, …, `d_II,N31 = 1.95` |
| \(r\) | – | – | `r = 0.60` |
| \(\tau\) | – | – | `τ = 0.10` |

All indices \(s\) and \(n\) are taken from the full sets of schools and neighborhoods present in the data; no additional filtering is applied. The model uses the exact numeric values as supplied in the files.