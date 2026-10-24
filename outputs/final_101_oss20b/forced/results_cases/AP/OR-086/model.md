**Mathematical model**

Let  

* \(G=\{I,II,III\}\) – raw‑grade indices  
* \(B=\{Red,Yellow,Blue\}\) – wine‑brand indices  

Decision variables  

\[
x_{g,b}\;\;(\text{kg})\qquad g\in G,\;b\in B
\]

Parameters  

\[
\begin{aligned}
c_g &\;\;(\text{CNY/kg}) &&\text{unit cost of grade }g\\
s_g &\;\;(\text{kg}) &&\text{daily supply limit of grade }g\\
p_b &\;\;(\text{CNY/kg}) &&\text{selling price of brand }b\\
\ell_{g,b} &\;\;(0\le \ell_{g,b}\le 1) &&\text{lower bound on the fraction of grade }g\text{ in brand }b\\
u_{g,b} &\;\;(0\le u_{g,b}\le 1) &&\text{upper bound on the fraction of grade }g\text{ in brand }b
\end{aligned}
\]

---

### Objective  

\[
\max \;\; \sum_{b\in B} p_b \sum_{g\in G} x_{g,b}
          \;-\; \sum_{g\in G} c_g \sum_{b\in B} x_{g,b}
\]

---

### Constraints  

1. **Blending‑proportion bounds** (for every \(b\in B,\;g\in G\))

\[
\ell_{g,b}\;\sum_{h\in G}x_{h,b}\;\le\;x_{g,b}\;\le\;
u_{g,b}\;\sum_{h\in G}x_{h,b}
\]

2. **Raw‑material supply limits** (for every \(g\in G\))

\[
\sum_{b\in B}x_{g,b}\;\le\; s_g
\]

3. **Minimum production of Red**  

\[
\sum_{g\in G}x_{g,Red}\;\ge\;2000
\]

4. **Non‑negativity**

\[
x_{g,b}\;\ge\;0\qquad \forall g\in G,\;b\in B
\]

---

## Data‑mapping rules

| Parameter | Source table | Source column | Mapping rule |
|-----------|--------------|---------------|--------------|
| \(c_g\) | `file_0_view_0` | `Cost (CNY/kg)` | indexed by `Grade` |
| \(s_g\) | `file_0_view_0` | `Daily Supply (kg)` | indexed by `Grade` |
| \(p_b\) | `file_1_view_0` | `Selling Price (CNY/kg)` | indexed by `Brand` |
| \(\ell_{g,b}\) | `file_1_view_0` | `Blending Requirements` | parse each requirement string: if it contains “more than X%” for grade \(g\) in brand \(b\), set \(\ell_{g,b}=X/100\); otherwise \(\ell_{g,b}=0\) |
| \(u_{g,b}\) | `file_1_view_0` | `Blending Requirements` | parse each requirement string: if it contains “less than X%” for grade \(g\) in brand \(b\), set \(u_{g,b}=X/100\); otherwise \(u_{g,b}=1\) |

*All other \(\ell_{g,b}\) and \(u_{g,b}\) values not specified in the requirement text are set to 0 and 1 respectively, allowing the corresponding grade to be absent or unrestricted in that brand.*