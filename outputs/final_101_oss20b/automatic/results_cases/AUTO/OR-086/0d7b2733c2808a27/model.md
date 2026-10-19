[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to build a linear blending model that decides how many kilograms of each raw wine grade (I, II, III) to allocate to each of the three wine brands (Red, Yellow, Blue) so that the daily net profit (sales revenue minus raw‑material cost) is maximised.  The model must respect:  
   * the upper‑ and lower‑bound proportion requirements for selected grades in each brand,  
   * the daily supply limits of each raw grade,  
   * a minimum daily production of 2 000 kg for the Red brand.  

2.  **Identify Model Type:** This is a **continuous linear programming (LP) blending problem** – all decision variables are non‑negative real numbers and the objective and constraints are linear.

3.  **Define Index Sets:**  
   * **G** – set of raw grades: {I, II, III}.  
   * **B** – set of wine brands: {Red, Yellow, Blue}.

4.  **Define Decision Variables:**  
   * `x[g,b]` = kilograms of raw grade `g` used in brand `b`.  
     Type: `GRB.CONTINUOUS`, `x[g,b] ≥ 0`.  

5.  **Identify Parameters (from Schema):**  
   * **Supply limits** – `Supply[g]` (kg) from column *Daily Supply (kg)* in 30‑1.csv.  
   * **Raw‑material costs** – `Cost[g]` (CNY/kg) from column *Cost (CNY/kg)* in 30‑1.csv.  
   * **Selling prices** – `Price[b]` (CNY/kg) from column *Selling Price (CNY/kg)* in 30‑2.csv.  
   * **Blending bounds** – for each brand `b` and grade `g` that appears in the *Blending Requirements* text, two parameters:  
     * `Lower[b,g]` – minimum proportion (0–1) of grade `g` in brand `b`.  
     * `Upper[b,g]` – maximum proportion (0–1) of grade `g` in brand `b`.  
     (Grades not mentioned for a brand have no bounds.)

6.  **Formulate Objective:**  
   Maximise total net profit  
   \[
   \max \;\sum_{b\in B}\;\Bigl( Price[b]\;\sum_{g\in G} x[g,b] \Bigr)
          \;-\;\sum_{g\in G}\;\Bigl( Cost[g]\;\sum_{b\in B} x[g,b] \Bigr).
   \]
   (Equivalently, \(\max \sum_{b,g} (Price[b]-Cost[g])\,x[g,b]\).)

7.  **Formulate Constraints:**  
   * **Raw‑material supply limits** – for each grade  
     \[
     \sum_{b\in B} x[g,b] \;\le\; Supply[g], \qquad \forall g\in G.
     \]
   * **Minimum production for Red** – let \(P_b = \sum_{g\in G} x[g,b]\) be the total output of brand \(b\)  
     \[
     P_{\text{Red}} \;\ge\; 2000.
     \]
   * **Blending proportion bounds** – for every brand–grade pair that has a bound  
     \[
     Lower[b,g]\;P_b \;\le\; x[g,b] \;\le\; Upper[b,g]\;P_b, \qquad \forall b\in B,\; \forall g\in G\text{ with bounds}.
     \]
   * **Non‑negativity** – \(x[g,b] \ge 0\) for all \(g,b\).

All constraints are linear because the right‑hand side of each proportion bound is a linear combination of the same variables \(x[g,b]\).  

This abstract plan fully specifies the LP model that can be implemented in any optimisation solver.  
[Abstract Model Plan END]