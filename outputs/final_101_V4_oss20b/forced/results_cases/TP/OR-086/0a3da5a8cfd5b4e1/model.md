**Mathematical model**

Let  

* \(G=\{I,II,III\}\) – raw‑grade indices  
* \(B=\{Red,Yellow,Blue\}\) – wine‑brand indices  

Decision variables  
\[
x_{g,b}\;\ge 0\qquad\forall g\in G,\;b\in B
\]
(kg of grade \(g\) used for brand \(b\))

Parameters  

| Symbol | Meaning | Source |
|--------|---------|--------|
| \(s_g\) | Daily supply limit of grade \(g\) (kg) | `file_0_view_0` – “Daily Supply (kg)” |
| \(c_g\) | Unit cost of grade \(g\) (CNY/kg) | `file_0_view_0` – “Cost (CNY/kg)” |
| \(p_b\) | Selling price of brand \(b\) (CNY/kg) | `file_1_view_0` – “Selling Price (CNY/kg)” |
| \(\ell_{g,b}\) | Lower proportion bound of grade \(g\) in brand \(b\) (fraction) | Derived from “Blending Requirements” |
| \(u_{g,b}\) | Upper proportion bound of grade \(g\) in brand \(b\) (fraction) | Derived from “Blending Requirements” |

(If a grade has no bound for a brand, set \(\ell_{g,b}=0,\;u_{g,b}=1\).)

Objective (maximise net profit)

\[
\max \;\; \sum_{b\in B} p_b \Bigl(\sum_{g\in G} x_{g,b}\Bigr)
          \;-\;
          \sum_{g\in G} c_g \Bigl(\sum_{b\in B} x_{g,b}\Bigr)
\]

Constraints  

1. **Raw‑material supply limits**  
   \[
   \sum_{b\in B} x_{g,b}\;\le\; s_g \qquad\forall g\in G
   \]

2. **Blending proportion bounds**  
   \[
   \ell_{g,b}\;\Bigl(\sum_{g'\in G} x_{g',b}\Bigr)
   \;\le\; x_{g,b}
   \;\le\;
   u_{g,b}\;\Bigl(\sum_{g'\in G} x_{g',b}\Bigr)
   \qquad\forall g\in G,\;b\in B
   \]

3. **Minimum production of Red**  
   \[
   \sum_{g\in G} x_{g,Red}\;\ge\; 2000
   \]

4. **Non‑negativity**  
   \[
   x_{g,b}\;\ge\;0 \qquad\forall g\in G,\;b\in B
   \]

---

**Data‑mapping rules**

| Parameter | Table ID | Column(s) | Mapping rule |
|-----------|----------|-----------|--------------|
| \(s_g\) | `file_0_view_0` | `Grade`, `Daily Supply (kg)` | For each row, set \(s_{\text{Grade}} = \text{Daily Supply (kg)}\). |
| \(c_g\) | `file_0_view_0` | `Grade`, `Cost (CNY/kg)` | For each row, set \(c_{\text{Grade}} = \text{Cost (CNY/kg)}\). |
| \(p_b\) | `file_1_view_0` | `Brand`, `Selling Price (CNY/kg)` | For each row, set \(p_{\text{Brand}} = \text{Selling Price (CNY/kg)}\). |
| \(\ell_{g,b}, u_{g,b}\) | `file_1_view_0` | `Brand`, `Blending Requirements` | Parse the text in `Blending Requirements` for each brand: for each clause “\<Grade\> less than \<x\>%” set \(u_{g,b}=x/100\); for “\<Grade\> more than \<x\>%” set \(\ell_{g,b}=x/100\). If a grade is not mentioned, set \(\ell_{g,b}=0,\;u_{g,b}=1\). |

These mappings provide all numerical values needed to instantiate the model. No additional data are required.