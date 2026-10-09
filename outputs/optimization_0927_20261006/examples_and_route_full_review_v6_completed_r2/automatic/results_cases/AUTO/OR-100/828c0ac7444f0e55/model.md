Let $x_i$ be the number of units to produce of component $i$ ($i \in \{C1, C2, \ldots, C111\}$), where $x_i \in \mathbb{Z}_{\geq 0}$.

Let $p_i$ be the unit price of component $i$ (from unit_price.csv).

Let $a_{wi}$ be the unit processing time required for component $i$ in workshop $w$ (from processing_time_unit.csv, $w \in \{\text{Casting}, \text{Milling}, \text{Finishing}, \text{Assembly}, \text{QA \& Packaging}\}$).

Let $b_w$ be the total available working hours in workshop $w$ (from total_working_hours.csv).

The model is:

Maximize total output value:
$$
\max \sum_{i=1}^{111} p_i x_i
$$

Subject to workshop capacity constraints (for each $w$):

$$
\sum_{i=1}^{111} a_{wi} x_i \leq b_w \qquad \forall w \in \{\text{Casting}, \text{Milling}, \text{Finishing}, \text{Assembly}, \text{QA \& Packaging}\}
$$

Nonnegativity and integrality:
$$
x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{C1, C2, \ldots, C111\}
$$

---

#### Data

**Component indices:** $i \in \{C1, C2, \ldots, C111\}$

**Workshops:** $w \in \{\text{Casting}, \text{Milling}, \text{Finishing}, \text{Assembly}, \text{QA \& Packaging}\}$

**Unit prices $p_i$ (from unit_price.csv, in source order):**
- $p_{C1} = 193$
- $p_{C2} = 64$
- $p_{C3} = 103$
- $p_{C4} = 210$
- $p_{C5} = 85$
- $p_{C6} = 126$
- $p_{C7} = 226$
- $p_{C8} = 94$
- $p_{C9} = 73$
- $p_{C10} = 120$
- $p_{C11} = 81$
- $p_{C12} = 94$
- $p_{C13} = 133$
- $p_{C14} = 197$
- $p_{C15} = 63$
- $p_{C16} = 159$
- $p_{C17} = 160$
- $p_{C18} = 97$
- $p_{C19} = 182$
- $p_{C20} = 128$
- $p_{C21} = 181$
- $p_{C22} = 171$
- $p_{C23} = 91$
- $p_{C24} = 228$
- $p_{C25} = 152$
- $p_{C26} = 85$
- $p_{C27} = 203$
- $p_{C28} = 134$
- $p_{C29} = 232$
- $p_{C30} = 125$
- $p_{C31} = 181$
- $p_{C32} = 246$
- $p_{C33} = 226$
- $p_{C34} = 88$
- $p_{C35} = 187$
- $p_{C36} = 152$
- $p_{C37} = 130$
- $p_{C38} = 86$
- $p_{C39} = 50$
- $p_{C40} = 229$
- $p_{C41} = 93$
- $p_{C42} = 169$
- $p_{C43} = 72$
- $p_{C44} = 67$
- $p_{C45} = 136$
- $p_{C46} = 118$
- $p_{C47} = 101$
- $p_{C48} = 94$
- $p_{C49} = 78$
- $p_{C50} = 76$
- $p_{C51} = 155$
- $p_{C52} = 114$
- $p_{C53} = 225$
- $p_{C54} = 238$
- $p_{C55} = 59$
- $p_{C56} = 135$
- $p_{C57} = 245$
- $p_{C58} = 231$
- $p_{C59} = 219$
- $p_{C60} = 167$
- $p_{C61} = 164$
- $p_{C62} = 139$
- $p_{C63} = 220$
- $p_{C64} = 167$
- $p_{C65} = 240$
- $p_{C66} = 170$
- $p_{C67} = 91$
- $p_{C68} = 106$
- $p_{C69} = 135$
- $p_{C70} = 91$
- $p_{C71} = 51$
- $p_{C72} = 73$
- $p_{C73} = 211$
- $p_{C74} = 189$
- $p_{C75} = 169$
- $p_{C76} = 153$
- $p_{C77} = 151$
- $p_{C78} = 167$
- $p_{C79} = 173$
- $p_{C80} = 152$
- $p_{C81} = 101$
- $p_{C82} = 216$
- $p_{C83} = 196$
- $p_{C84} = 92$
- $p_{C85} = 92$
- $p_{C86} = 97$
- $p_{C87} = 224$
- $p_{C88} = 128$
- $p_{C89} = 139$
- $p_{C90} = 109$
- $p_{C91} = 206$
- $p_{C92} = 161$
- $p_{C93} = 227$
- $p_{C94} = 187$
- $p_{C95} = 106$
- $p_{C96} = 248$
- $p_{C97} = 82$
- $p_{C98} = 222$
- $p_{C99} = 209$
- $p_{C100} = 223$
- $p_{C101} = 204$
- $p_{C102} = 114$
- $p_{C103} = 146$
- $p_{C104} = 231$
- $p_{C105} = 93$
- $p_{C106} = 224$
- $p_{C107} = 220$
- $p_{C108} = 100$
- $p_{C109} = 187$
- $p_{C110} = 213$
- $p_{C111} = 142$

**Unit processing times $a_{wi}$ (from processing_time_unit.csv, in source order):**

- For each workshop $w$ (Casting, Milling, Finishing, Assembly, QA & Packaging), and each component $i$ ($C1$ to $C111$), $a_{wi}$ is the value in the corresponding row and column.

**Total available working hours $b_w$ (from total_working_hours.csv):**
- $b_{\text{Casting}} = 7650$
- $b_{\text{Milling}} = 6320$
- $b_{\text{Finishing}} = 5538$
- $b_{\text{Assembly}} = 5957$
- $b_{\text{QA \& Packaging}} = 6988$