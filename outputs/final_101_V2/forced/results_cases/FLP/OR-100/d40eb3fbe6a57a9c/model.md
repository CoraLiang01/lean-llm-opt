##### Decision Variables

Let $x_i \geq 0$ denote the production quantity of component $i$, for $i = 1,2,\ldots,111$ (corresponding to $C1, C2, \ldots, C111$).

##### Parameters

Let $p_i$ be the unit price of component $i$ (from unit_price.csv).

Let $a_{wi}$ be the unit processing time required for component $i$ in workshop $w$ (from processing_time_unit.csv), where $w \in \{\text{Casting}, \text{Milling}, \text{Finishing}, \text{Assembly}, \text{QA \& Packaging}\}$.

Let $T_w$ be the total available working hours for workshop $w$ (from total_working_hours.csv).

All parameter values are as follows:

- Components: $C1, C2, \ldots, C111$
- Workshops: Casting, Milling, Finishing, Assembly, QA & Packaging

**Unit Prices ($p_i$):**

| Component | Unit Price |
|-----------|------------|
| C1   | 193 | C2   | 64  | C3   | 103 | C4   | 210 | C5   | 85  | C6   | 126 | C7   | 226 | C8   | 94  | C9   | 73  | C10  | 120 |
| C11  | 81  | C12  | 94  | C13  | 133 | C14  | 197 | C15  | 63  | C16  | 159 | C17  | 160 | C18  | 97  | C19  | 182 | C20  | 128 |
| C21  | 181 | C22  | 171 | C23  | 91  | C24  | 228 | C25  | 152 | C26  | 85  | C27  | 203 | C28  | 134 | C29  | 232 | C30  | 125 |
| C31  | 181 | C32  | 246 | C33  | 226 | C34  | 88  | C35  | 187 | C36  | 152 | C37  | 130 | C38  | 86  | C39  | 50  | C40  | 229 |
| C41  | 93  | C42  | 169 | C43  | 72  | C44  | 67  | C45  | 136 | C46  | 118 | C47  | 101 | C48  | 94  | C49  | 78  | C50  | 76  |
| C51  | 155 | C52  | 114 | C53  | 225 | C54  | 238 | C55  | 59  | C56  | 135 | C57  | 245 | C58  | 231 | C59  | 219 | C60  | 167 |
| C61  | 164 | C62  | 139 | C63  | 220 | C64  | 167 | C65  | 240 | C66  | 170 | C67  | 91  | C68  | 106 | C69  | 135 | C70  | 91  |
| C71  | 51  | C72  | 73  | C73  | 211 | C74  | 189 | C75  | 169 | C76  | 153 | C77  | 151 | C78  | 167 | C79  | 173 | C80  | 152 |
| C81  | 101 | C82  | 216 | C83  | 196 | C84  | 92  | C85  | 92  | C86  | 97  | C87  | 224 | C88  | 128 | C89  | 139 | C90  | 109 |
| C91  | 206 | C92  | 161 | C93  | 227 | C94  | 187 | C95  | 106 | C96  | 248 | C97  | 82  | C98  | 222 | C99  | 209 | C100 | 223 |
| C101 | 204 | C102 | 114 | C103 | 146 | C104 | 231 | C105 | 93  | C106 | 224 | C107 | 220 | C108 | 100 | C109 | 187 | C110 | 213 |
| C111 | 142 |

**Unit Processing Times ($a_{wi}$):**

For each component $i$ and workshop $w$, $a_{wi}$ is given in the table below (see processing_time_unit.csv for all values). For example:

| Component | Casting | Milling | Finishing | Assembly | QA & Packaging |
|-----------|---------|---------|-----------|----------|---------------|
| C1   | 0.74 | 0.6  | 0.0  | 4.84 | 0.92 |
| C2   | 0.77 | 3.38 | 4.15 | 0.0  | 0.0  |
| ...  | ...  | ...  | ...  | ...  | ...  |
| C111| 3.81 | 2.67 | 4.07 | 4.02 | 3.21 |

**Total Available Working Hours ($T_w$):**

| Workshop           | Total Hours |
|--------------------|------------|
| Casting            | 7650       |
| Milling            | 6320       |
| Finishing          | 5538       |
| Assembly           | 5957       |
| QA & Packaging     | 6988       |

##### Objective Function

\[
\max \sum_{i=1}^{111} p_i x_i
\]

##### Constraints

For each workshop $w$ (Casting, Milling, Finishing, Assembly, QA & Packaging):

\[
\sum_{i=1}^{111} a_{wi} x_i \leq T_w
\]

where $a_{wi}$ is the unit processing time of component $i$ in workshop $w$, and $T_w$ is the total available working hours for workshop $w$.

##### Variable Domains

\[
x_i \geq 0 \quad \text{for all } i = 1,2,\ldots,111
\]

##### Complete Mathematical Model

\[
\begin{align*}
\max \quad & \sum_{i=1}^{111} p_i x_i \\
\text{s.t.} \quad & \sum_{i=1}^{111} a_{wi} x_i \leq T_w, \quad \forall w \in \{\text{Casting}, \text{Milling}, \text{Finishing}, \text{Assembly}, \text{QA \& Packaging}\} \\
& x_i \geq 0, \quad \forall i = 1,2,\ldots,111
\end{align*}
\]

where all $p_i$, $a_{wi}$, and $T_w$ are as listed above and in the retrieved tables.