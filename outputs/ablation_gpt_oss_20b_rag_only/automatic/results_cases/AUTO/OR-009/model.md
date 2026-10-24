**Mathematical Model**

\[
\begin{aligned}
\min\;&\sum_{s\in S}\sum_{c\in C} \; \text{cost}_{s,c}\;x_{s,c} \\[4pt]
\text{s.t.}\;&\sum_{s\in S} x_{s,c}= \text{demand}_{c}\qquad \forall\,c\in C \\[4pt]
&\sum_{c\in C} x_{s,c}\le \text{capacity}_{s}\qquad \forall\,s\in S \\[4pt]
&x_{s,c}\;\ge\;0\qquad \forall\,s\in S,\;c\in C
\end{aligned}
\]

*Sets*  
\(S=\{S1,S2,S3,S4\}\) (plant identifiers)  
\(C=\{C1,C2,C3,C4\}\) (retail outlet identifiers)

*Parameters*  
\(\text{demand}_{c}\) daily demand of outlet \(c\)  
\(\text{capacity}_{s}\) production capacity of plant \(s\)  
\(\text{cost}_{s,c}\) unit transportation cost from plant \(s\) to outlet \(c\)

*Decision variables*  
\(x_{s,c}\) quantity shipped from plant \(s\) to outlet \(c\) (continuous, \(\ge 0\))

---

**Data Mapping Rules**

| Parameter | Source Table | Column(s) | Mapping Notes |
|-----------|--------------|-----------|---------------|
| \(\text{demand}_{c}\) | `file_0_view_0` | `customer` → index, `demand` → value | One row per customer; `customer` values are the set \(C\). |
| \(\text{capacity}_{s}\) | `file_1_view_0` | `Unnamed: 0` → index, `supply_capacity` → value | One row per plant; `Unnamed: 0` values are the set \(S\). |
| \(\text{cost}_{s,c}\) | `file_2_view_0` | `Unnamed: 0` → plant index, columns `C1`–`C4` → costs | Matrix of size \(|S|\times|C|\); row and column identifiers align exactly with the sets \(S\) and \(C\). |

All indices and values are taken verbatim from the returned rows of the respective CSV views; no additional aggregation or transformation is performed. The model uses the full set of entities present in the observation.