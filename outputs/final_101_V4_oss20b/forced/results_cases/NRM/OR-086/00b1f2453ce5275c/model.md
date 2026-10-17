**Mathematical Model**

Let  

* \(G=\{I,II,III\}\) : raw‑grade indices  
* \(B=\{Red,Yellow,Blue\}\) : wine‑brand indices  

**Parameters**  

\[
\begin{aligned}
&\text{Supply}_g &&\text{(kg)} &&\text{from file_0_view_0[Daily Supply (kg)]} \\
&\text{Cost}_g &&\text{(CNY/kg)} &&\text{from file_0_view_0[Cost (CNY/kg)]} \\
&\text{Price}_b &&\text{(CNY/kg)} &&\text{from file_1_view_0[Selling Price (CNY/kg)]} \\
&\underline{L}_{b,g} &&\text{(fraction)} &&\text{lower bound on grade }g\text{ in brand }b \\
&\overline{U}_{b,g} &&\text{(fraction)} &&\text{upper bound on grade }g\text{ in brand }b
\end{aligned}
\]

(For a brand–grade pair that has no stated bound, the corresponding parameter is omitted; no constraint is imposed.)

**Decision Variables**  

\[
x_{b,g}\;\;(\text{kg}) \quad\ge 0
\]

Quantity of raw grade \(g\) used in brand \(b\).

**Auxiliary Variables**  

\[
T_b = \sum_{g\in G} x_{b,g}\qquad(\text{kg of brand }b\text{ produced})
\]

**Objective**  

\[
\max \;\; \sum_{b\in B}\text{Price}_b\,T_b \;-\; \sum_{g\in G}\text{Cost}_g\sum_{b\in B}x_{b,g}
\]

**Constraints**

1. **Blending proportions**  
   \[
   \begin{aligned}
   &x_{b,g} \;\ge\; \underline{L}_{b,g}\,T_b &&\forall\,b\in B,\;g\in G\;\text{with a lower bound} \\
   &x_{b,g} \;\le\; \overline{U}_{b,g}\,T_b &&\forall\,b\in B,\;g\in G\;\text{with an upper bound}
   \end{aligned}
   \]

2. **Raw‑material supply**  
   \[
   \sum_{b\in B} x_{b,g}\;\le\;\text{Supply}_g \qquad\forall\,g\in G
   \]

3. **Minimum production of Red**  
   \[
   T_{\text{Red}}\;\ge\;2000
   \]

4. **Definition of total production**  
   \[
   T_b = \sum_{g\in G} x_{b,g}\qquad\forall\,b\in B
   \]

All variables \(x_{b,g}\) are continuous and non‑negative.

---

### Data Mapping Rules

| Parameter | Source Table | Source Column | Mapping Rule |
|-----------|--------------|---------------|--------------|
| \(\text{Supply}_g\) | `file_0_view_0` | `Daily Supply (kg)` | For each record where `Grade = g`, assign the value to \(\text{Supply}_g\). |
| \(\text{Cost}_g\) | `file_0_view_0` | `Cost (CNY/kg)` | For each record where `Grade = g`, assign the value to \(\text{Cost}_g\). |
| \(\text{Price}_b\) | `file_1_view_0` | `Selling Price (CNY/kg)` | For each record where `Brand = b`, assign the value to \(\text{Price}_b\). |
| \(\underline{L}_{b,g}\) | `file_1_view_0` | `Blending Requirements` | Parse the string for brand \(b\); for each clause “\(g\) more than \(p\)%”, set \(\underline{L}_{b,g}=p/100\). |
| \(\overline{U}_{b,g}\) | `file_1_view_0` | `Blending Requirements` | Parse the string for brand \(b\); for each clause “\(g\) less than \(p\)%”, set \(\overline{U}_{b,g}=p/100\). |

No other data are required. The model uses only the parameters mapped above and the decision variables defined.