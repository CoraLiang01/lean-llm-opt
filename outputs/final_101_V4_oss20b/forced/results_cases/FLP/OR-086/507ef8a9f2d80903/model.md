**Symbolic Model**

\[
\begin{aligned}
\text{Sets } & G=\{I,II,III\}, \qquad B=\{\text{Red},\text{Yellow},\text{Blue}\} \\[4pt]
\text{Parameters } & 
\begin{cases}
s_g &\text{daily supply limit of grade }g \\
c_g &\text{unit cost of grade }g \\
p_b &\text{selling price of brand }b \\
l_{b,g} &\text{lower proportion bound of grade }g\text{ in brand }b \\
u_{b,g} &\text{upper proportion bound of grade }g\text{ in brand }b
\end{cases} \\[4pt]
\text{Decision variables } & 
\begin{cases}
x_{b,g}\ge 0 &\text{kg of grade }g\text{ used in brand }b \\
X_b=\sum_{g\in G}x_{b,g} &\text{total production of brand }b
\end{cases} \\[4pt]
\text{Objective } & 
\max \;\; \sum_{b\in B} p_b\,X_b \;-\; \sum_{g\in G} c_g\!\!\sum_{b\in B}x_{b,g} \\[6pt]
\text{Constraints } & 
\begin{aligned}
&\text{(1) Blending bounds: } && l_{b,g}\,X_b \;\le\; x_{b,g}\;\le\; u_{b,g}\,X_b &&\forall b\in B,\;g\in G \\[4pt]
&\text{(2) Supply limits: } && \sum_{b\in B}x_{b,g}\;\le\; s_g &&\forall g\in G \\[4pt]
&\text{(3) Minimum production: } && X_{\text{Red}}\;\ge\;2000
\end{aligned}
\end{aligned}
\]

---

**Data‑Mapping Rules**

| Parameter | Source Table | Column | Notes |
|-----------|--------------|--------|-------|
| \(s_g\) | `file_0_view_0` | `Daily Supply (kg)` | one value per grade |
| \(c_g\) | `file_0_view_0` | `Cost (CNY/kg)` | one value per grade |
| \(p_b\) | `file_1_view_0` | `Selling Price (CNY/kg)` | one value per brand |
| \(l_{b,g}\) | `file_1_view_0` | `Blending Requirements` | parsed from text; if no lower bound for \((b,g)\) set to 0 |
| \(u_{b,g}\) | `file_1_view_0` | `Blending Requirements` | parsed from text; if no upper bound for \((b,g)\) set to 1 |

*Parsing rule for `Blending Requirements`*  
Each entry contains phrases of the form “\<Grade\> less than \<percent\>” or “\<Grade\> more than \<percent\>”.  
- “less than X%” → \(u_{b,g}=X/100\).  
- “more than X%” → \(l_{b,g}=X/100\).  
If a grade is mentioned only with one bound, the other bound defaults to 0 (lower) or 1 (upper).  

All other parameters are taken directly from the corresponding columns without modification.