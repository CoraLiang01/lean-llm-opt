**Symbolic Model**

Let  

* \(S\) – set of schools (from *school_capacity.csv*).  
* \(N\) – set of neighborhoods (from *neighborhoods_population.csv*).  

Parameters  

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(c_s\) | Capacity of school \(s\) | `school_capacity.csv` column *Capacity* |
| \(p^{W}_n\) | White‑student population in neighborhood \(n\) | `neighborhoods_population.csv` column *Population_White* |
| \(p^{NW}_n\) | Non‑white‑student population in neighborhood \(n\) | `neighborhoods_population.csv` column *Population_NonWhite* |
| \(d_{sn}\) | Distance (miles) from school \(s\) to neighborhood \(n\) | `distance.csv` columns *N01*…*N31* |

Decision variables  

* \(x^{W}_{sn}\ge 0\) – number of white students from neighborhood \(n\) assigned to school \(s\).  
* \(x^{NW}_{sn}\ge 0\) – number of non‑white students from neighborhood \(n\) assigned to school \(s\).

Objective  

\[
\min \sum_{s\in S}\sum_{n\in N} d_{sn}\,\bigl(x^{W}_{sn}+x^{NW}_{sn}\bigr)
\]

Constraints  

1. **Neighborhood population balance**  
   \[
   \sum_{s\in S} x^{W}_{sn}=p^{W}_n,\qquad
   \sum_{s\in S} x^{NW}_{sn}=p^{NW}_n,\qquad \forall n\in N
   \]

2. **School capacity**  
   \[
   \sum_{n\in N}\bigl(x^{W}_{sn}+x^{NW}_{sn}\bigr)\le c_s,\qquad \forall s\in S
   \]

3. **Racial‑balance (within ±10 % of 60 % white)**  
   Let \(T_{sn}=x^{W}_{sn}+x^{NW}_{sn}\).  
   \[
   0.5\,\sum_{n\in N}T_{sn}\;\le\;\sum_{n\in N}x^{W}_{sn}\;\le\;0.7\,\sum_{n\in N}T_{sn},
   \qquad \forall s\in S
   \]

4. **Non‑negativity**  
   \[
   x^{W}_{sn}\ge0,\;x^{NW}_{sn}\ge0,\qquad \forall s\in S,\;n\in N
   \]

---

**Data‑Mapping Rules**

| Parameter | Table ID | Column |
|-----------|----------|--------|
| \(c_s\) | `file_0_view_0` | `Capacity` |
| \(p^{W}_n\) | `file_1_view_0` | `Population_White` |
| \(p^{NW}_n\) | `file_1_view_0` | `Population_NonWhite` |
| \(d_{sn}\) | `file_2_view_0` | column named after neighborhood (e.g., `N01`, `N02`, …, `N31`) |

All index sets \(S\) and \(N\) are taken from the distinct values of the corresponding identifier columns in the source tables. No additional numerical values are introduced beyond those supplied in the CSV files.