**Mathematical model**

\[
\begin{aligned}
\text{Maximize}\quad
Z \;=&\; \sum_{b\in B}\;p_b\;\sum_{g\in G}x_{g,b}
      \;-\;\sum_{g\in G}\;c_g\;\sum_{b\in B}x_{g,b} \\[4pt]
\text{subject to}\quad
&\text{(Blending bounds)}\qquad
   \ell_{g,b}\;\sum_{h\in G}x_{h,b}\;\le\;x_{g,b}\;\le\;
   u_{g,b}\;\sum_{h\in G}x_{h,b}\;\;\;\forall g\in G,\;b\in B \\[4pt]
&\text{(Supply limits)}\qquad
   \sum_{b\in B}x_{g,b}\;\le\;S_g\;\;\;\forall g\in G \\[4pt]
&\text{(Minimum Red production)}\qquad
   \sum_{g\in G}x_{g,\text{Red}}\;\ge\;2000 \\[4pt]
&x_{g,b}\;\ge\;0\;\;\;\forall g\in G,\;b\in B
\end{aligned}
\]

*Indices*  
\(G=\{I,\,II,\,III\}\) (grade of raw wine)  
\(B=\{\text{Red},\,\text{Yellow},\,\text{Blue}\}\) (wine brand)

*Parameters*  

| Symbol | Meaning | Source table & column |
|--------|---------|-----------------------|
| \(S_g\) | Daily supply limit of grade \(g\) (kg) | `file_0_view_0`, column **Daily Supply (kg)** |
| \(c_g\) | Unit cost of grade \(g\) (CNY/kg) | `file_0_view_0`, column **Cost (CNY/kg)** |
| \(p_b\) | Selling price of brand \(b\) (CNY/kg) | `file_1_view_0`, column **Selling Price (CNY/kg)** |
| \(\ell_{g,b}\) | Lower‑bound proportion of grade \(g\) in brand \(b\) | Derived by parsing **Blending Requirements** in `file_1_view_0` |
| \(u_{g,b}\) | Upper‑bound proportion of grade \(g\) in brand \(b\) | Derived by parsing **Blending Requirements** in `file_1_view_0` |

*Notes on blending bounds*  
- If a grade is not mentioned in the “Blending Requirements” string for a brand, the corresponding bound is taken as \(\ell_{g,b}=0\) and \(u_{g,b}=1\).  
- The bounds are fractions (e.g., “I less than 10%” → \(u_{I,\text{Red}}=0.10\); “II more than 50%” → \(\ell_{II,\text{Red}}=0.50\)).

*Decision variables*  

\(x_{g,b}\) – quantity (kg) of raw grade \(g\) allocated to brand \(b\).  
\(x_{g,b}\ge 0\).

---

**Data‑mapping rules**

| Parameter | Table ID | Column name | Mapping rule |
|-----------|----------|-------------|--------------|
| \(S_g\) | `file_0_view_0` | `Daily Supply (kg)` | `S_g = value of column "Daily Supply (kg)" for grade g` |
| \(c_g\) | `file_0_view_0` | `Cost (CNY/kg)` | `c_g = value of column "Cost (CNY/kg)" for grade g` |
| \(p_b\) | `file_1_view_0` | `Selling Price (CNY/kg)` | `p_b = value of column "Selling Price (CNY/kg)" for brand b` |
| \(\ell_{g,b}\), \(u_{g,b}\) | `file_1_view_0` | `Blending Requirements` | Parse the string for brand b; for each grade g extract the numeric bound and assign to \(\ell_{g,b}\) or \(u_{g,b}\) accordingly; if absent, set \(\ell_{g,b}=0\), \(u_{g,b}=1\) |

These mappings provide the complete numerical data needed to instantiate the model. No additional data are required.