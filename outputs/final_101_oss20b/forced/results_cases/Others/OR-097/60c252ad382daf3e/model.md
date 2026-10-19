[Abstract Model Plan START]
1.  **Analyze Query:** The user wants to determine integer option positions \(x_i\) (positive for long, negative for short) that minimize the total hedging cost while ensuring that the portfolio’s net Delta, Gamma, and Vega exposures lie within specified tolerance bands. The solution must respect per‑option trading limits and use the asset‑reference information to compute each option’s contribution to the Greeks.

2.  **Identify Model Type:** Mixed‑Integer Linear Programming (MIP) with absolute‑value linearization (to handle \(|x_i|\) in the objective).

3.  **Define Index Sets:**  
   - \(I = \{1,\dots,120\}\) – set of option contracts.  
   - \(A = \{1,\dots,6\}\) – set of underlying assets.

4.  **Define Decision Variables:**  
   - `x[i]` = integer number of contracts of option \(i\) (positive = long, negative = short).  
     Type: `GRB.INTEGER`.  
   - `u[i]` = auxiliary non‑negative integer representing \(|x_i|\).  
     Type: `GRB.INTEGER`.

5.  **Identify Parameters (from Schema):**  
   - `Cost[i]` – per‑contract hedging cost (from *OptionCharacteristics.csv*).  
   - `Delta[i]`, `Gamma[i]`, `Vega[i]` – per‑contract Greek exposures (from *OptionCharacteristics.csv*).  
   - `MaxLong[i]`, `MaxShort[i]` – per‑option trading limits (from *OptionCharacteristics.csv*).  
   - `A[i,j]` – binary asset‑reference matrix (from *Option_AssetReferenceMatrix.csv*).  
   - `Delta_init = 0.25`, `Gamma_init = 0.08`, `Vega_init = 0.17` – initial portfolio exposures.  
   - `Tol_Delta = 0.06`, `Tol_Gamma = 0.05`, `Tol_Vega = 0.07` – tolerance bands.

6.  **Formulate Objective:**  
   Minimize total hedging cost  
   \[
   \min \sum_{i\in I} \text{Cost}[i] \cdot u[i]
   \]
   subject to the linearization constraints that enforce \(u[i] \ge |x_i|\).

7.  **Formulate Constraints:**  
   - **Absolute‑value linearization (for each \(i\in I\)):**  
     \[
     \begin{aligned}
     u[i] &\ge x[i] \\
     u[i] &\ge -x[i] \\
     u[i] &\ge 0
     \end{aligned}
     \]
   - **Delta exposure band:**  
     \[
     \begin{aligned}
     \Delta_{\text{init}} + \sum_{i\in I}\sum_{j\in A} \Delta[i]\;A[i,j]\;x[i] &\le \text{Tol}_{\Delta} \\
     -\Bigl(\Delta_{\text{init}} + \sum_{i\in I}\sum_{j\in A} \Delta[i]\;A[i,j]\;x[i]\Bigr) &\le \text{Tol}_{\Delta}
     \end{aligned}
     \]
   - **Gamma exposure band:**  
     \[
     \begin{aligned}
     \Gamma_{\text{init}} + \sum_{i\in I}\sum_{j\in A} \Gamma[i]\;A[i,j]\;x[i] &\le \text{Tol}_{\Gamma} \\
     -\Bigl(\Gamma_{\text{init}} + \sum_{i\in I}\sum_{j\in A} \Gamma[i]\;A[i,j]\;x[i]\Bigr) &\le \text{Tol}_{\Gamma}
     \end{aligned}
     \]
   - **Vega exposure band:**  
     \[
     \begin{aligned}
     \text{Vega}_{\text{init}} + \sum_{i\in I}\sum_{j\in A} \text{Vega}[i]\;A[i,j]\;x[i] &\le \text{Tol}_{\text{Vega} } \\
     -\Bigl(\text{Vega}_{\text{init}} + \sum_{i\in I}\sum_{j\in A} \text{Vega}[i]\;A[i,j]\;x[i]\Bigr) &\le \text{Tol}_{\text{Vega}}
     \end{aligned}
     \]
   - **Trading limits (for each \(i\in I\)):**  
     \[
     \text{MaxShort}[i] \;\le\; x[i] \;\le\; \text{MaxLong}[i]
     \]
   - **Integrality:**  
     \[
     x[i] \in \mathbb{Z}, \quad u[i] \in \mathbb{Z}_{\ge 0} \quad \forall i\in I
     \]

These constraints fully capture the user’s requirements: minimizing hedging cost, keeping net Greeks within tolerance bands, respecting per‑option limits, and ensuring integer contract positions.  
[Abstract Model Plan END]