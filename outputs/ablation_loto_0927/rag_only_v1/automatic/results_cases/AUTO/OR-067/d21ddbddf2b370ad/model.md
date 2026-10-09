Let there be three managers: MA, MB, and MC, and three projects: P1, P2, and P3. Let the decision variable \( x_{ij} \) be defined as:

\[
x_{ij} = 
\begin{cases}
1 & \text{if manager } i \text{ is assigned to project } j \\
0 & \text{otherwise}
\end{cases}
\]

where \( i \in \{\text{MA}, \text{MB}, \text{MC}\} \) and \( j \in \{\text{P1}, \text{P2}, \text{P3}\} \).

The cost matrix \( C \) is:

\[
C = \begin{bmatrix}
3000 & 3200 & 3100 \\
2800 & 3300 & 2900 \\
2900 & 3100 & 3000 \\
\end{bmatrix}
\]

where the rows correspond to MA, MB, MC and the columns correspond to P1, P2, P3, respectively.

The mathematical model is:

\[
\begin{align*}
\text{Minimize} \quad & Z = 3000x_{\text{MA},\text{P1}} + 3200x_{\text{MA},\text{P2}} + 3100x_{\text{MA},\text{P3}} \\
&\quad\quad\quad + 2800x_{\text{MB},\text{P1}} + 3300x_{\text{MB},\text{P2}} + 2900x_{\text{MB},\text{P3}} \\
&\quad\quad\quad + 2900x_{\text{MC},\text{P1}} + 3100x_{\text{MC},\text{P2}} + 3000x_{\text{MC},\text{P3}} \\
\text{subject to:} \\
& x_{\text{MA},\text{P1}} + x_{\text{MA},\text{P2}} + x_{\text{MA},\text{P3}} = 1 \quad \text{(MA assigned to one project)} \\
& x_{\text{MB},\text{P1}} + x_{\text{MB},\text{P2}} + x_{\text{MB},\text{P3}} = 1 \quad \text{(MB assigned to one project)} \\
& x_{\text{MC},\text{P1}} + x_{\text{MC},\text{P2}} + x_{\text{MC},\text{P3}} = 1 \quad \text{(MC assigned to one project)} \\
& x_{\text{MA},\text{P1}} + x_{\text{MB},\text{P1}} + x_{\text{MC},\text{P1}} = 1 \quad \text{(P1 assigned to one manager)} \\
& x_{\text{MA},\text{P2}} + x_{\text{MB},\text{P2}} + x_{\text{MC},\text{P2}} = 1 \quad \text{(P2 assigned to one manager)} \\
& x_{\text{MA},\text{P3}} + x_{\text{MB},\text{P3}} + x_{\text{MC},\text{P3}} = 1 \quad \text{(P3 assigned to one manager)} \\
& x_{ij} \in \{0,1\} \quad \forall i \in \{\text{MA}, \text{MB}, \text{MC}\},\ j \in \{\text{P1}, \text{P2}, \text{P3}\}
\end{align*}
\]

Where:
- The objective function minimizes the total cost of assigning managers to projects, using the actual costs from the CSV.
- The first three constraints ensure each manager is assigned to exactly one project.
- The next three constraints ensure each project is assigned to exactly one manager.
- All variables are binary.

This is the complete mathematical model for the optimal assignment of managers to projects, using the provided cost data.