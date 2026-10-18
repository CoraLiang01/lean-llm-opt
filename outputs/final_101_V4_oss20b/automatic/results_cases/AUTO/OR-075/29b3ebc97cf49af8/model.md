[Abstract Model Plan START]  
1. **Analyze Query:** The user wants to select a subset of 110 potential investment projects that maximizes the total expected Net Present Value (NPV) while respecting a capital budget of 1,000 k$ and three strategic dependency constraints:  
   - Only one of Project 4 (R&D Initiative Alpha) or Project 7 (Global Expansion Pilot) may be chosen.  
   - Project 6 (System Automation) can be selected only if Project 1 (Infrastructure Upgrade) is also selected.  
   - Project 10 (Customer Experience Platform) can be selected only if Project 5 (Staff Training Program) is also selected.  

2. **Identify Model Type:** This is a **Mixed Integer Programming (MIP)** problem with binary decision variables (project selection).  

3. **Define Index Sets:**  
   - `P` : set of all projects, indexed by project ID (1 … 110).  

4. **Define Decision Variables:**  
   - `x[p]` = 1 if project *p* is selected, 0 otherwise.  
     Type: `GRB.BINARY`.  

5. **Identify Parameters (from Schema):**  
   - `Capital[p]` : capital requirement of project *p* (k$), from column **Capital (k$)**.  
   - `NPV[p]` : expected net present value of project *p* (k$), from column **NPV (k$)**.  
   - Budget limit `B = 1000` (k$).  

6. **Formulate Objective:**  
   - Maximize total NPV:  
     \[
     \max \sum_{p \in P} \text{NPV}[p] \cdot x[p]
     \]  

7. **Formulate Constraints:**  
   - **Budget Constraint:**  
     \[
     \sum_{p \in P} \text{Capital}[p] \cdot x[p] \;\le\; B
     \]  
   - **Mutually Exclusive Constraint (Projects 4 & 7):**  
     \[
     x[4] + x[7] \;\le\; 1
     \]  
   - **Prerequisite Constraint (Project 6 requires 1):**  
     \[
     x[6] \;\le\; x[1]
     \]  
   - **Contingent Constraint (Project 10 requires 5):**  
     \[
     x[10] \;\le\; x[5]
     \]  

All other variables are binary, and no additional constraints are required.  

[Abstract Model Plan END]