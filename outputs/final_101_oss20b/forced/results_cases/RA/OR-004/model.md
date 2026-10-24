**Mathematical Model**

\[
\begin{aligned}
\text{Minimize}\quad & \sum_{s\in S}\sum_{c\in C} \text{cost}_{s,c}\;x_{s,c} \\[4pt]
\text{subject to}\quad
& \sum_{s\in S} x_{s,c} \;\ge\; \text{demand}_{c} \qquad &&\forall\,c\in C \quad\text{(Demand)}\\[4pt]
& \sum_{c\in C} x_{s,c} \;\le\; \text{supply}_{s} \qquad &&\forall\,s\in S \quad\text{(Supply)}\\[4pt]
& x_{s,c}\;\in\;\mathbb{Z}_{\ge 0} \qquad &&\forall\,s\in S,\;c\in C
\end{aligned}
\]

---

**Data Mapping**

| Parameter | Table ID | Row Key | Column Key |
|-----------|----------|---------|------------|
| \(\text{demand}_{c}\) | `file_0_view_0` | `customer` | `demand` |
| \(\text{supply}_{s}\) | `file_1_view_0` | `Unnamed: 0` | `supply_capacity` |
| \(\text{cost}_{s,c}\) | `file_2_view_0` | `Unnamed: 0` | `C1` … `C12` |

*Index sets*  
\(S = \{\,s \mid s \text{ appears in } \text{file\_1\_view\_0.Unnamed: 0}\,\}\)  
\(C = \{\,c \mid c \text{ appears in } \text{file\_0\_view\_0.customer}\,\}\)

All parameters are non‑negative real numbers; decision variables \(x_{s,c}\) are non‑negative integers.