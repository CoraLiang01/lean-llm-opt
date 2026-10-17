**Symbolic Model**

\[
\begin{aligned}
\text{Minimize}\quad
& Z = \sum_{s\in S} f_s\,y_s \;+\; \sum_{s\in S}\sum_{c\in C} t_{s,c}\,x_{s,c} \\[4pt]
\text{subject to}\quad
& \sum_{s\in S} x_{s,c} \;\ge\; d_c , && \forall\,c\in C \qquad\text{(demand)}\\[4pt]
& x_{s,c} \;\le\; d_c\,y_s , && \forall\,s\in S,\;c\in C \qquad\text{(linking)}\\[4pt]
& y_s \in \{0,1\}, && \forall\,s\in S \\[2pt]
& x_{s,c} \in \mathbb{Z}_{\ge 0}, && \forall\,s\in S,\;c\in C
\end{aligned}
\]

*Index sets*  
\(S=\{S1,S2,S3,S4,S5,S6\}\) (suppliers)  
\(C=\{C1,C2,C3,C4,C5,C6\}\) (customers)

*Parameters*  
\(f_s\) fixed cost of opening supplier \(s\)  
\(d_c\) demand of customer \(c\)  
\(t_{s,c}\) transportation cost per unit from supplier \(s\) to customer \(c\)

*Variables*  
\(y_s\) binary, 1 if supplier \(s\) is opened  
\(x_{s,c}\) integer quantity shipped from \(s\) to \(c\)

---

**Data Mapping**

| Parameter | Source Table | Source Column |
|-----------|--------------|---------------|
| \(f_s\)   | `file_1_view_0` | `fixed_costs` |
| \(d_c\)   | `file_0_view_0` | `demand` |
| \(t_{s,c}\) | `file_2_view_0` | `C1`–`C6` (columns corresponding to customers) |

All indices \(s\) and \(c\) are taken from the row and column identifiers of the respective tables. No additional data are introduced beyond what is supplied in the observation.