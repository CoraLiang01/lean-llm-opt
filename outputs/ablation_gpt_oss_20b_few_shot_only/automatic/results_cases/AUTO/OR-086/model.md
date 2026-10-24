**Mathematical model**

Let  

\[
G=\{I,\,II,\,III\},\qquad B=\{\text{Red},\,\text{Yellow},\,\text{Blue}\}
\]

Parameters  

\[
\begin{aligned}
S_g &\;\;=\;\text{daily supply of grade }g\quad(\text{kg})\\
C_g &\;\;=\;\text{unit cost of grade }g\quad(\text{CNY/kg})\\
P_b &\;\;=\;\text{selling price of brand }b\quad(\text{CNY/kg})\\
L_{g,b}&\;\;=\;\text{lower proportion bound of grade }g\text{ in brand }b\\
U_{g,b}&\;\;=\;\text{upper proportion bound of grade }g\text{ in brand }b
\end{aligned}
\]

Decision variables  

\[
x_{g,b}\;\ge 0\quad\text{(kg of grade }g\text{ used in brand }b)
\]

Objective  

\[
\max Z=\sum_{g\in G}\sum_{b\in B}\bigl(P_b-C_g\bigr)\,x_{g,b}
\]

Constraints  

1. **Raw‑material supply**  
   \[
   \sum_{b\in B}x_{g,b}\;\le\;S_g\qquad\forall g\in G
   \]

2. **Minimum production for Red**  
   \[
   \sum_{g\in G}x_{g,\text{Red}}\;\ge\;2000
   \]

3. **Blending proportions**  
   Let \(Q_b=\sum_{h\in G}x_{h,b}\) (total production of brand \(b\)).  
   \[
   L_{g,b}\,Q_b\;\le\;x_{g,b}\;\le\;U_{g,b}\,Q_b
   \qquad\forall g\in G,\;\forall b\in B
   \]

   (If a grade has no stated requirement for a brand, set \(L_{g,b}=0,\;U_{g,b}=1\).)

---

**Data‑mapping rules**

| File | Column | Parameter | Notes |
|------|--------|-----------|-------|
| `Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-1.csv` | `Grade` | index \(g\) | values: I, II, III |
| | `Daily Supply (kg)` | \(S_g\) | supply limit for each grade |
| | `Cost (CNY/kg)` | \(C_g\) | unit cost for each grade |
| `Test_Dataset/Large-scale-or/Mixture_testing/Mixture8/30-2.csv` | `Brand` | index \(b\) | values: Red, Yellow, Blue |
| | `Selling Price (CNY/kg)` | \(P_b\) | selling price for each brand |
| | `Blending Requirements` | defines \(L_{g,b}\) and \(U_{g,b}\) | parse phrases:<br>• “I less than 10%” → \(U_{I,b}=0.10\)<br>• “II more than 50%” → \(L_{II,b}=0.50\)<br>• “III less than 70%” → \(U_{III,b}=0.70\)<br>• “I more than 20%” → \(L_{I,b}=0.20\)<br>• “I less than 50%” → \(U_{I,b}=0.50\)<br>• “II more than 10%” → \(L_{II,b}=0.10\) |

All other \(L_{g,b}\) and \(U_{g,b}\) not mentioned above are set to 0 and 1, respectively. The minimum‑production constraint applies only to the Red brand. All variables \(x_{g,b}\) are continuous and non‑negative.