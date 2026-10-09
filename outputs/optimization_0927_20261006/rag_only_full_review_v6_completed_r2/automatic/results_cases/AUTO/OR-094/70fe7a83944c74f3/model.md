Let:
- \( x_j \) = number of units of radio model HiFi\(j\) to produce per day, for \( j = 1, \ldots, 101 \). Each \( x_j \) is a nonnegative integer.

Let:
- \( t_{i,j} \) = processing time (in minutes) required per unit of model HiFi\(j\) at workstation \(i\), for \( i = 1,2,3 \), as given in the data.
- \( C_i \) = effective daily capacity (in minutes) of workstation \(i\), after maintenance.

From the data:
- Each workstation has 1,440 minutes per day.
- Maintenance percentages: workstation 1: 10%, workstation 2: 14%, workstation 3: 12%.

Thus,
- \( C_1 = 1440 \times (1 - 0.10) = 1296 \) minutes
- \( C_2 = 1440 \times (1 - 0.14) = 1238.4 \) minutes
- \( C_3 = 1440 \times (1 - 0.12) = 1267.2 \) minutes

Let the set of models be \( J = \{1,2,\ldots,101\} \).

Let the processing times \( t_{i,j} \) be as follows (from the CSV, for each workstation \(i\), the value for HiFi\(j\)_Minutes):

- For workstation 1 (i=1):  
  \( t_{1,1} = 6 \), \( t_{1,2} = 4 \), ..., \( t_{1,101} = 9 \)
- For workstation 2 (i=2):  
  \( t_{2,1} = 5 \), \( t_{2,2} = 5 \), ..., \( t_{2,101} = 3 \)
- For workstation 3 (i=3):  
  \( t_{3,1} = 4 \), \( t_{3,2} = 6 \), ..., \( t_{3,101} = 6 \)

Define the idle time at each workstation as:
- \( \text{Idle}_i = C_i - \sum_{j=1}^{101} t_{i,j} x_j \), for \( i = 1,2,3 \)

Objective:
Minimize total idle time across all workstations:
\[
\min \sum_{i=1}^3 \left( C_i - \sum_{j=1}^{101} t_{i,j} x_j \right)
\]
which is equivalent to:
\[
\max \sum_{i=1}^3 \sum_{j=1}^{101} t_{i,j} x_j
\]
since the \( C_i \) are constants, but per the user query, the objective must be to minimize total idle time (not maximize production), so we keep the original form.

Constraints:
For each workstation, total processing time used cannot exceed its effective capacity:
\[
\sum_{j=1}^{101} t_{i,j} x_j \leq C_i \quad \forall i = 1,2,3
\]

Variable domains:
\[
x_j \in \mathbb{Z}_{\geq 0} \quad \forall j = 1,2,\ldots,101
\]

Numerical Formulation:

Let the models be indexed as follows (in the order from the CSV columns):

\[
\begin{align*}
\text{Let } x_j &= \text{number of units of HiFi}j \text{ to produce, for } j=1,\ldots,101 \\
\text{Let } t_{1,j} &= \text{HiFi}j\_\text{Minutes} \text{ from row Workstation=1} \\
\text{Let } t_{2,j} &= \text{HiFi}j\_\text{Minutes} \text{ from row Workstation=2} \\
\text{Let } t_{3,j} &= \text{HiFi}j\_\text{Minutes} \text{ from row Workstation=3} \\
\end{align*}
\]

Objective:
\[
\min \left[ (1296 - \sum_{j=1}^{101} t_{1,j} x_j) + (1238.4 - \sum_{j=1}^{101} t_{2,j} x_j) + (1267.2 - \sum_{j=1}^{101} t_{3,j} x_j) \right]
\]

Constraints:
\[
\sum_{j=1}^{101} t_{1,j} x_j \leq 1296
\]
\[
\sum_{j=1}^{101} t_{2,j} x_j \leq 1238.4
\]
\[
\sum_{j=1}^{101} t_{3,j} x_j \leq 1267.2
\]

\[
x_j \in \mathbb{Z}_{\geq 0} \quad \forall j=1,\ldots,101
\]

Where the coefficients \( t_{i,j} \) are as follows (from the CSV, in order):

- For \( j = 1 \) (HiFi1): \( t_{1,1} = 6 \), \( t_{2,1} = 5 \), \( t_{3,1} = 4 \)
- For \( j = 2 \) (HiFi2): \( t_{1,2} = 4 \), \( t_{2,2} = 5 \), \( t_{3,2} = 6 \)
- ...
- For \( j = 101 \) (HiFi101): \( t_{1,101} = 9 \), \( t_{2,101} = 3 \), \( t_{3,101} = 6 \)

All data is taken directly from the supplied CSV rows and columns, in order.

Summary Table of Variables and Coefficients (partial, for illustration):

| Model (j) | \( t_{1,j} \) | \( t_{2,j} \) | \( t_{3,j} \) |
|-----------|--------------|--------------|--------------|
| HiFi1     | 6            | 5            | 4            |
| HiFi2     | 4            | 5            | 6            |
| ...       | ...          | ...          | ...          |
| HiFi101   | 9            | 3            | 6            |

Decision variables: \( x_j \) for \( j = 1, \ldots, 101 \), all nonnegative integers.

This model minimizes the total idle production time across all three workstations, subject to the effective daily capacity of each workstation and the per-unit processing times for each radio model at each workstation.