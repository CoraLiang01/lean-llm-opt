**Mathematical Model**

\[
\begin{aligned}
\min\;&\sum_{s\in S}\sum_{c\in C} \; \text{cost}_{s,c}\;x_{s,c} \\[4pt]
\text{s.t.}\;&\sum_{c\in C} x_{s,c}\;\le\;\text{supply}_{s}\qquad \forall s\in S \\[4pt]
&\sum_{s\in S} x_{s,c}\;=\;\text{demand}_{c}\qquad \forall c\in C \\[4pt]
&x_{s,c}\;\ge\;0\qquad \forall s\in S,\;c\in C
\end{aligned}
\]

*Sets*  
\(S=\{S1,S2,\dots ,S12\}\) (Distribution centers)  
\(C=\{C1,C2,\dots ,C12\}\) (Customer groups)

*Parameters*  
\(\text{demand}_{c}\) – daily demand of customer \(c\)  
\(\text{supply}_{s}\) – supply capacity of center \(s\)  
\(\text{cost}_{s,c}\) – transportation cost per unit from \(s\) to \(c\)

*Decision variables*  
\(x_{s,c}\) – units shipped from center \(s\) to customer \(c\)

---

**Data‑Mapping Rules**

| Parameter | Source Table | Source Columns | Mapping |
|-----------|--------------|----------------|---------|
| \(\text{demand}_{c}\) | `file_0_view_0` | `customer` → `c`, `demand` → `demand` | `demand[c]` |
| \(\text{supply}_{s}\) | `file_1_view_0` | `Unnamed: 0` → `s`, `supply_capacity` → `supply` | `supply[s]` |
| \(\text{cost}_{s,c}\) | `file_2_view_0` | `Unnamed: 0` → `s`, `C1`–`C12` → `c1`–`c12` | `cost[s,c]` |

*Note:* The column names `C1`–`C12` in `file_2_view_0` correspond directly to the customer identifiers `C1`–`C12`. The mapping preserves the exact identifiers and does not enumerate numerical values.