**Mathematical Model**

Let  

- \(S\) be the set of Walmart stores (indices from *file_1_view_0*).  
- \(C\) be the set of customer groups (indices from *file_0_view_0*).  

Parameters  

\[
\begin{aligned}
&\text{demand}[c] &&\text{(units required by customer }c)\\
&\text{supply}[s] &&\text{(daily supply capacity of store }s)\\
&\text{cost}[s,c] &&\text{(transportation cost per unit from store }s\text{ to customer }c)
\end{aligned}
\]

Decision variables  

\[
x_{s,c}\in \mathbb{Z}_{\ge 0}\qquad \forall s\in S,\;c\in C
\]

Objective  

\[
\min \;\sum_{s\in S}\sum_{c\in C}\text{cost}[s,c]\;x_{s,c}
\]

Constraints  

1. **Demand satisfaction**  
\[
\sum_{s\in S} x_{s,c}\;\ge\;\text{demand}[c]\qquad \forall c\in C
\]

2. **Supply capacity**  
\[
\sum_{c\in C} x_{s,c}\;\le\;\text{supply}[s]\qquad \forall s\in S
\]

3. **Non‑negativity & integrality**  
\[
x_{s,c}\in \mathbb{Z}_{\ge 0}\qquad \forall s\in S,\;c\in C
\]

---

**Data Mapping**

| Parameter | Source Table | Column(s) |
|-----------|--------------|-----------|
| \(S\) | *file_1_view_0* | `Unnamed: 0` |
| \(C\) | *file_0_view_0* | `customer` |
| `demand[c]` | *file_0_view_0* | `demand` |
| `supply[s]` | *file_1_view_0* | `supply_capacity` |
| `cost[s,c]` | *file_2_view_0* | `C1` … `C12` (columns aligned with customers) |

All indices and parameters are bound directly to the corresponding rows and columns of the supplied CSV views. No additional data are introduced.