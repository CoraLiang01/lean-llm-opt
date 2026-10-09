**Parameters:**

- Components $k \in \{\text{C1}, \text{C2}, \ldots, \text{C111}\}$
- Workshops $w \in \{\text{Casting}, \text{Milling}, \text{Finishing}, \text{Assembly}, \text{QA \& Packaging}\}$

- $p_k$ (unit price):

| Component | unit_price |
|-----------|------------|
| C1        | 193        |
| C2        | 64         |
| C3        | 103        |
| C4        | 210        |
| C5        | 85         |
| C6        | 126        |
| C7        | 226        |
| C8        | 94         |
| C9        | 73         |
| C10       | 120        |
| C11       | 81         |
| C12       | 94         |
| C13       | 133        |
| C14       | 197        |
| C15       | 63         |
| C16       | 159        |
| C17       | 160        |
| C18       | 97         |
| C19       | 182        |
| C20       | 128        |
| C21       | 181        |
| C22       | 171        |
| C23       | 91         |
| C24       | 228        |
| C25       | 152        |
| C26       | 85         |
| C27       | 203        |
| C28       | 134        |
| C29       | 232        |
| C30       | 125        |
| C31       | 181        |
| C32       | 246        |
| C33       | 226        |
| C34       | 88         |
| C35       | 187        |
| C36       | 152        |
| C37       | 130        |
| C38       | 86         |
| C39       | 50         |
| C40       | 229        |
| C41       | 93         |
| C42       | 169        |
| C43       | 72         |
| C44       | 67         |
| C45       | 136        |
| C46       | 118        |
| C47       | 101        |
| C48       | 94         |
| C49       | 78         |
| C50       | 76         |
| C51       | 155        |
| C52       | 114        |
| C53       | 225        |
| C54       | 238        |
| C55       | 59         |
| C56       | 135        |
| C57       | 245        |
| C58       | 231        |
| C59       | 219        |
| C60       | 167        |
| C61       | 164        |
| C62       | 139        |
| C63       | 220        |
| C64       | 167        |
| C65       | 240        |
| C66       | 170        |
| C67       | 91         |
| C68       | 106        |
| C69       | 135        |
| C70       | 91         |
| C71       | 51         |
| C72       | 73         |
| C73       | 211        |
| C74       | 189        |
| C75       | 169        |
| C76       | 153        |
| C77       | 151        |
| C78       | 167        |
| C79       | 173        |
| C80       | 152        |
| C81       | 101        |
| C82       | 216        |
| C83       | 196        |
| C84       | 92         |
| C85       | 92         |
| C86       | 97         |
| C87       | 224        |
| C88       | 128        |
| C89       | 139        |
| C90       | 109        |
| C91       | 206        |
| C92       | 161        |
| C93       | 227        |
| C94       | 187        |
| C95       | 106        |
| C96       | 248        |
| C97       | 82         |
| C98       | 222        |
| C99       | 209        |
| C100      | 223        |
| C101      | 204        |
| C102      | 114        |
| C103      | 146        |
| C104      | 231        |
| C105      | 93         |
| C106      | 224        |
| C107      | 220        |
| C108      | 100        |
| C109      | 187        |
| C110      | 213        |
| C111      | 142        |

- $a_{w,k}$ (processing time per unit in workshop $w$ for component $k$):

| Workshop           | C1   | C2   | ... | C111 |
|--------------------|------|------|-----|------|
| Casting            | 0.74 | 0.77 | ... | 3.81 |
| Milling            | 0.6  | 3.38 | ... | 2.67 |
| Finishing          | 0.0  | 4.15 | ... | 4.07 |
| Assembly           | 4.84 | 0.0  | ... | 4.02 |
| QA & Packaging     | 0.92 | 0.0  | ... | 3.21 |

(Full table as in processing_time_unit.csv, all 5 workshops × 111 components.)

- $T_w$ (total available working hours):

| Workshop           | total_hours |
|--------------------|-------------|
| Casting            | 7650        |
| Milling            | 6320        |
| Finishing          | 5538        |
| Assembly           | 5957        |
| QA & Packaging     | 6988        |

**Decision Variables:**

- $x_k \in \mathbb{Z}_{\geq 0}$, for all $k \in \{\text{C1}, \ldots, \text{C111}\}$

**Mathematical Model:**

**Objective:**
\[
\max \sum_{k \in \{\text{C1}, \ldots, \text{C111}\}} p_k \cdot x_k
\]

**Subject to:**

For each workshop $w$:
\[
\sum_{k \in \{\text{C1}, \ldots, \text{C111}\}} a_{w,k} \cdot x_k \leq T_w
\]
That is,
- **Casting:** $\sum_{k} a_{\text{Casting},k} \cdot x_k \leq 7650$
- **Milling:** $\sum_{k} a_{\text{Milling},k} \cdot x_k \leq 6320$
- **Finishing:** $\sum_{k} a_{\text{Finishing},k} \cdot x_k \leq 5538$
- **Assembly:** $\sum_{k} a_{\text{Assembly},k} \cdot x_k \leq 5957$
- **QA & Packaging:** $\sum_{k} a_{\text{QA \& Packaging},k} \cdot x_k \leq 6988$

**Variable domains:**
\[
x_k \in \mathbb{Z}_{\geq 0}, \quad \forall k \in \{\text{C1}, \ldots, \text{C111}\}
\]

**Where:**
- $p_k$ is the unit price of component $k$ (see table above)
- $a_{w,k}$ is the processing time per unit of component $k$ in workshop $w$ (see processing_time_unit.csv)
- $T_w$ is the total available working hours in workshop $w$ (see table above)

**All coefficients and identifiers are as in the original data.**