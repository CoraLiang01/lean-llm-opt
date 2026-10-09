##### Sets

- $M = \{\text{MA}, \text{MB}, \text{MC}\}$ (managers, in source order)
- $P = \{\text{P1}, \text{P2}, \text{P3}\}$ (projects, in source order)

##### Parameters

- $c_{mp}$: cost for manager $m$ to complete project $p$, as given below:

\[
\begin{array}{c|ccc}
 & \text{P1} & \text{P2} & \text{P3} \\
\hline
\text{MA} & 3000 & 3200 & 3100 \\
\text{MB} & 2800 & 3300 & 2900 \\
\text{MC} & 2900 & 3100 & 3000 \\
\end{array}
\]

##### Decision Variables

- $x_{mp} \in \{0,1\}$: $1$ if manager $m$ is assigned to project $p$, $0$ otherwise.

##### Objective

\[
\min \sum_{m \in M} \sum_{p \in P} c_{mp} x_{mp}
\]

##### Constraints

1. Each manager is assigned to exactly one project:
   \[
   \sum_{p \in P} x_{mp} = 1 \quad \forall m \in M
   \]

2. Each project is assigned to exactly one manager:
   \[
   \sum_{m \in M} x_{mp} = 1 \quad \forall p \in P
   \]

3. Binary assignment:
   \[
   x_{mp} \in \{0,1\} \quad \forall m \in M,\, p \in P
   \]

##### Data (from manager_project_costs.csv, source order)

- Managers: MA, MB, MC
- Projects: P1, P2, P3
- Costs:
    - MA: P1 = 3000, P2 = 3200, P3 = 3100
    - MB: P1 = 2800, P2 = 3300, P3 = 2900
    - MC: P1 = 2900, P2 = 3100, P3 = 3000