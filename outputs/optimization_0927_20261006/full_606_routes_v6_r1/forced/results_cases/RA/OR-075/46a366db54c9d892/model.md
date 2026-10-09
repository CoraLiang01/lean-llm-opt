Let $x_i$ be a binary variable indicating whether Project $i$ is selected ($x_i = 1$) or not ($x_i = 0$), for each project with "Project ID" $i$ from 1 to 110.

**Parameters (from project.csv, in source order):**

| Project ID | Project Name                      | Capital (k$) | NPV (k$) |
|------------|-----------------------------------|--------------|----------|
| 1          | Infrastructure Upgrade            | 50           | 60       |
| 2          | New Product Line A                | 40           | 50       |
| 3          | Marketing Campaign X              | 30           | 45       |
| 4          | R&D Initiative Alpha              | 25           | 35       |
| 5          | Staff Training Program            | 20           | 28       |
| 6          | System Automation                 | 65           | 75       |
| 7          | Global Expansion Pilot            | 80           | 100      |
| 8          | Green Energy Switch               | 15           | 20       |
| 9          | Warehouse Optimization            | 48           | 62       |
| 10         | Customer Experience Platform      | 55           | 70       |
| 11         | Project 011 - Strategy Focus      | 87           | 117      |
| 12         | Project 012 - Strategy Focus      | 82           | 109      |
| 13         | Project 013 - Strategy Focus      | 123          | 172      |
| 14         | Project 014 - Strategy Focus      | 137          | 180      |
| 15         | Project 015 - Strategy Focus      | 43           | 62       |
| 16         | Project 016 - Strategy Focus      | 110          | 147      |
| 17         | Project 017 - Strategy Focus      | 94           | 141      |
| 18         | Project 018 - Strategy Focus      | 101          | 151      |
| 19         | Project 019 - Strategy Focus      | 54           | 75       |
| 20         | Project 020 - Strategy Focus      | 143          | 185      |
| ...        | ...                               | ...          | ...      |
| 110        | Project 110 - Strategy Focus      | 127          | 177      |

(Full data for all 110 projects is included as above.)

---

### Mathematical Model

**Decision Variables:**

- $x_i \in \{0,1\}$ for $i = 1,2,\ldots,110$ (1 if project $i$ is selected, 0 otherwise)

---

**Objective:**

$$
\max \sum_{i=1}^{110} \text{NPV}_i \cdot x_i
$$

where $\text{NPV}_i$ is the "NPV (k$)" for project $i$.

---

**Constraints:**

1. **Budget Constraint:**
   $$
   \sum_{i=1}^{110} \text{Capital}_i \cdot x_i \leq 1000
   $$
   where $\text{Capital}_i$ is the "Capital (k$)" for project $i$.

2. **Mutually Exclusive Constraint (Projects 4 & 7):**
   $$
   x_4 + x_7 \leq 1
   $$

3. **Pre-requisite Constraint (Project 6 requires 1):**
   $$
   x_6 \leq x_1
   $$

4. **Contingent Constraint (Project 10 requires 5):**
   $$
   x_{10} \leq x_5
   $$

5. **Binary Variables:**
   $$
   x_i \in \{0,1\} \quad \forall i = 1,2,\ldots,110
   $$

---

**All coefficients and project IDs are as given in the retrieved data, in source order.**