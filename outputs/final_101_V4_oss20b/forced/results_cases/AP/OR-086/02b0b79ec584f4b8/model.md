**Mathematical Model**

Let  

* \(G=\{I,II,III\}\) – raw‑grade indices  
* \(B=\{\text{Red},\text{Yellow},\text{Blue}\}\) – brand indices  

Decision variables  

\[
x_{g,b}\;\;(\text{kg}) \quad\forall g\in G,\;b\in B
\]

\[
y_b=\sum_{g\in G}x_{g,b}\qquad(\text{kg of brand }b)
\]

Parameters  

\[
\begin{aligned}
&\text{cost}_g &&\text{(CNY/kg)}\\
&\text{price}_b &&\text{(CNY/kg)}\\
&\text{supply}_g &&\text{(kg)}\\
&\underline{p}_{g,b} &&\text{lower proportion bound (fraction)}\\
&\overline{p}_{g,b} &&\text{upper proportion bound (fraction)}
\end{aligned}
\]

---

### Objective  

\[
\max \;\; \sum_{b\in B}\text{price}_b\,y_b \;-\; \sum_{g\in G}\text{cost}_g\sum_{b\in B}x_{g,b}
\]

---

### Constraints  

1. **Blending proportion bounds** (only for grades with specified bounds)  

\[
\underline{p}_{g,b}\,y_b \;\le\; x_{g,b} \;\le\; \overline{p}_{g,b}\,y_b
\qquad\forall g\in G,\;b\in B
\]

2. **Raw‑material supply limits**  

\[
\sum_{b\in B}x_{g,b}\;\le\;\text{supply}_g
\qquad\forall g\in G
\]

3. **Minimum production for Red**  

\[
y_{\text{Red}}\;\ge\;2000
\]

4. **Non‑negativity**  

\[
x_{g,b}\;\ge\;0 \qquad\forall g\in G,\;b\in B
\]

---

**Data Mapping**

| Parameter | Source | Table ID | Column(s) |
|-----------|--------|----------|-----------|
| \(\text{cost}_g\) | raw‑grade unit cost | `file_0_view_0` | `Cost (CNY/kg)` |
| \(\text{supply}_g\) | raw‑grade daily supply | `file_0_view_0` | `Daily Supply (kg)` |
| \(\text{price}_b\) | brand selling price | `file_1_view_0` | `Selling Price (CNY/kg)` |
| \(\underline{p}_{g,b}\) | lower blending proportion | `file_1_view_0` | parsed from `Blending Requirements` |
| \(\overline{p}_{g,b}\) | upper blending proportion | `file_1_view_0` | parsed from `Blending Requirements` |

*The blending proportion bounds are extracted from the textual `Blending Requirements` field for each brand, interpreting statements such as “I less than 10%” as \(\overline{p}_{I,b}=0.10\) and “II more than 50%” as \(\underline{p}_{II,b}=0.50\).*