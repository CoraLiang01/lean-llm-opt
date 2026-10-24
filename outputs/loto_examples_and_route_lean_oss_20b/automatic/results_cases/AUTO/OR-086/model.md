**Mathematical Model**

Let  

\[
\mathcal{G}=\{I,II,III\}, \qquad \mathcal{B}=\{\text{Red},\text{Yellow},\text{Blue}\}
\]

Decision variables  

\[
x_{g,b}\;\;(\text{kg of grade }g\text{ used in brand }b),\qquad g\in\mathcal{G},\,b\in\mathcal{B}
\]

Auxiliary variables  

\[
P_b=\sum_{g\in\mathcal{G}}x_{g,b}\qquad(b\in\mathcal{B})
\]

Parameters  

\[
\begin{aligned}
&\text{Supply}_g &&\text{(kg)}\\
&\text{Cost}_g &&\text{(CNY/kg)}\\
&\text{Price}_b &&\text{(CNY/kg)}\\
&\text{Lower}_{g,b},\;\text{Upper}_{g,b} &&\text{(proportion bounds, }0\le\text{Lower}\le\text{Upper}\le1)
\end{aligned}
\]

Objective  

\[
\max \;\sum_{b\in\mathcal{B}}\Bigl(\text{Price}_b\,P_b-\sum_{g\in\mathcal{G}}\text{Cost}_g\,x_{g,b}\Bigr)
\]

Constraints  

1. **Blending proportion bounds**  
   \[
   \begin{aligned}
   &\text{Lower}_{g,b}\,P_b \;\le\; x_{g,b}\;\le\;\text{Upper}_{g,b}\,P_b
   &&\forall\,g\in\mathcal{G},\,b\in\mathcal{B}\;\text{with a bound defined}
   \end{aligned}
   \]

2. **Raw‑material supply limits**  
   \[
   \sum_{b\in\mathcal{B}}x_{g,b}\;\le\;\text{Supply}_g
   &&\forall\,g\in\mathcal{G}
   \]

3. **Minimum production for Red**  
   \[
   P_{\text{Red}}\;\ge\;2000
   \]

4. **Non‑negativity**  
   \[
   x_{g,b}\;\ge\;0\qquad\forall\,g\in\mathcal{G},\,b\in\mathcal{B}
   \]

---

**Data Mapping**

| Parameter | Source Table | Column |
|-----------|--------------|--------|
| \(\mathcal{G}\) | `file_0_view_0` | `Grade` |
| \(\mathcal{B}\) | `file_1_view_0` | `Brand` |
| `Supply_g` | `file_0_view_0` | `Daily Supply (kg)` |
| `Cost_g` | `file_0_view_0` | `Cost (CNY/kg)` |
| `Price_b` | `file_1_view_0` | `Selling Price (CNY/kg)` |
| `Lower_{g,b}`, `Upper_{g,b}` | `file_1_view_0` | `Blending Requirements` (parsed text) |

*The blending bounds are extracted from the textual `Blending Requirements` column and mapped to the corresponding \((g,b)\) pairs.*