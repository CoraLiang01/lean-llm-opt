**Mathematical Model**

\[
\begin{aligned}
\text{Minimize}\quad & Z \;=\; \sum_{s\in S}\;\sum_{c\in C}\;c_{s,c}\,x_{s,c} \\[4pt]
\text{subject to}\quad
&\sum_{s\in S} x_{s,c}\;\ge\;d_c \qquad &&\forall\,c\in C \quad\text{(demand)}\\[4pt]
&\sum_{c\in C} x_{s,c}\;\le\;s_s \qquad &&\forall\,s\in S \quad\text{(supply)}\\[4pt]
&x_{s,c}\;\ge\;0 \qquad &&\forall\,s\in S,\;c\in C
\end{aligned}
\]

*Indices*  
\(S=\{\text{Supplier1},\dots,\text{Supplier5}\}\)  
\(C=\{\text{Customer1},\dots,\text{Customer6}\}\)

*Parameters*  
\(d_c\) – demand of customer \(c\)  
\(s_s\) – supply capacity of supplier \(s\)  
\(c_{s,c}\) – transportation cost per unit from supplier \(s\) to customer \(c\)

*Decision variables*  
\(x_{s,c}\) – amount of produce shipped from supplier \(s\) to customer \(c\)  
\(x_{s,c}\in\mathbb{R}_{+}\) (continuous non‑negative; integer if required)

---

**Data Mapping**

| Parameter | Table ID | Key Column | Value Column(s) | Notes |
|-----------|----------|------------|-----------------|-------|
| \(d_c\) | `file_0_view_0` | `Customers` | `demand` | One row per customer |
| \(s_s\) | `file_1_view_0` | `Suppliers` | `supply_capacity` | One row per supplier |
| \(c_{s,c}\) | `file_2_view_0` | `Unnamed: 0` (supplier) | `Customer1` … `Customer6` | Matrix of costs; each row corresponds to a supplier, each column to a customer |

All indices and parameter values are taken directly from the supplied CSV views; no additional data are introduced.