**Mathematical Model**

\[
\begin{aligned}
\text{Maximize}\quad
Z \;=&\; \sum_{b\in B}\;p_b\;\sum_{g\in G}x_{g,b}
      \;-\;\sum_{g\in G}\;c_g\;\sum_{b\in B}x_{g,b} \\[4pt]
\text{subject to}\quad
&\sum_{b\in B}x_{g,b}\;\le\;S_g \qquad &&\forall g\in G \quad\text{(Raw‑material supply)}\\[4pt]
&\ell_{g,b}\;\sum_{h\in G}x_{h,b}\;\le\;x_{g,b}\;\le\;u_{g,b}\;\sum_{h\in G}x_{h,b}
      &&\forall g\in G,\;\forall b\in B \quad\text{(Blending proportions)}\\[4pt]
&\sum_{g\in G}x_{g,\text{Red}}\;\ge\;2000 \qquad &&\text{(Red minimum production)}\\[4pt]
&x_{g,b}\;\ge\;0 \qquad &&\forall g\in G,\;\forall b\in B
\end{aligned}
\]

*Indices*  
\(G=\{I,\,II,\,III\}\) (grade of raw wine)  
\(B=\{\text{Red},\,\text{Yellow},\,\text{Blue}\}\) (wine brand)

*Parameters*  
\(S_g\) – daily supply limit of grade \(g\) (kg)  
\(c_g\) – unit cost of grade \(g\) (CNY/kg)  
\(p_b\) – selling price of brand \(b\) (CNY/kg)  
\(\ell_{g,b}\) – lower proportion bound of grade \(g\) in brand \(b\) (dimensionless)  
\(u_{g,b}\) – upper proportion bound of grade \(g\) in brand \(b\) (dimensionless)

*Decision variables*  
\(x_{g,b}\) – quantity of raw grade \(g\) allocated to brand \(b\) (kg)  
\(x_{g,b}\in\mathbb{R}_{+}\)

---

### Data Mapping

| Parameter | Source Table | Column(s) | Transformation |
|-----------|--------------|-----------|----------------|
| \(S_g\) | `file_0_view_0` | `Grade`, `Daily Supply (kg)` | Map `Grade` → \(g\); `Daily Supply (kg)` → \(S_g\) |
| \(c_g\) | `file_0_view_0` | `Grade`, `Cost (CNY/kg)` | Map `Grade` → \(g\); `Cost (CNY/kg)` → \(c_g\) |
| \(p_b\) | `file_1_view_0` | `Brand`, `Selling Price (CNY/kg)` | Map `Brand` → \(b\); `Selling Price (CNY/kg)` → \(p_b\) |
| \(\ell_{g,b}\), \(u_{g,b}\) | `file_1_view_0` | `Brand`, `Blending Requirements` | Parse the textual field `Blending Requirements` for each brand \(b\) to extract numeric bounds for each grade \(g\).  
  - If a statement “\(g\) less than \(x\%\)” → \(u_{g,b}=x/100\).  
  - If a statement “\(g\) more than \(y\%\)” → \(\ell_{g,b}=y/100\).  
  - Unmentioned grades receive \(\ell_{g,b}=0,\;u_{g,b}=1\). |
| Minimum production for Red | – | – | Fixed RHS 2000 kg for brand `Red` |

All other parameters are taken directly from the CSV files; no additional data are invented. The model uses continuous, non‑negative decision variables as specified.