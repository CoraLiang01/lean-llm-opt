Let $x_i$ be the number of units to produce of component $i$ ($i \in \{\text{C1}, \text{C2}, \ldots, \text{C111}\}$), where $x_i \in \mathbb{Z}_{\geq 0}$.

Let $p_i$ be the unit price of component $i$ (from unit_price.csv).

Let $a_{wi}$ be the unit processing time required for component $i$ in workshop $w$ (from processing_time_unit.csv, $w \in \{\text{Casting}, \text{Milling}, \text{Finishing}, \text{Assembly}, \text{QA \& Packaging}\}$).

Let $b_w$ be the total available working hours in workshop $w$ (from total_working_hours.csv).

The model is:

Maximize total output value:
$$
\max \sum_{i \in \{\text{C1}, \ldots, \text{C111}\}} p_i x_i
$$

Subject to workshop capacity constraints (for each $w$):

$$
\sum_{i \in \{\text{C1}, \ldots, \text{C111}\}} a_{wi} x_i \leq b_w \qquad \forall w \in \{\text{Casting}, \text{Milling}, \text{Finishing}, \text{Assembly}, \text{QA \& Packaging}\}
$$

and

$$
x_i \in \mathbb{Z}_{\geq 0} \qquad \forall i \in \{\text{C1}, \ldots, \text{C111}\}
$$

---

#### Data

**Workshops and total available working hours:**
- Casting: $b_{\text{Casting}} = 7650$
- Milling: $b_{\text{Milling}} = 6320$
- Finishing: $b_{\text{Finishing}} = 5538$
- Assembly: $b_{\text{Assembly}} = 5957$
- QA & Packaging: $b_{\text{QA \& Packaging}} = 6988$

**Unit prices:**
- $p_{\text{C1}} = 193$
- $p_{\text{C2}} = 64$
- $p_{\text{C3}} = 103$
- $p_{\text{C4}} = 210$
- $p_{\text{C5}} = 85$
- $p_{\text{C6}} = 126$
- $p_{\text{C7}} = 226$
- $p_{\text{C8}} = 94$
- $p_{\text{C9}} = 73$
- $p_{\text{C10}} = 120$
- $p_{\text{C11}} = 81$
- $p_{\text{C12}} = 94$
- $p_{\text{C13}} = 133$
- $p_{\text{C14}} = 197$
- $p_{\text{C15}} = 63$
- $p_{\text{C16}} = 159$
- $p_{\text{C17}} = 160$
- $p_{\text{C18}} = 97$
- $p_{\text{C19}} = 182$
- $p_{\text{C20}} = 128$
- $p_{\text{C21}} = 181$
- $p_{\text{C22}} = 171$
- $p_{\text{C23}} = 91$
- $p_{\text{C24}} = 228$
- $p_{\text{C25}} = 152$
- $p_{\text{C26}} = 85$
- $p_{\text{C27}} = 203$
- $p_{\text{C28}} = 134$
- $p_{\text{C29}} = 232$
- $p_{\text{C30}} = 125$
- $p_{\text{C31}} = 181$
- $p_{\text{C32}} = 246$
- $p_{\text{C33}} = 226$
- $p_{\text{C34}} = 88$
- $p_{\text{C35}} = 187$
- $p_{\text{C36}} = 152$
- $p_{\text{C37}} = 130$
- $p_{\text{C38}} = 86$
- $p_{\text{C39}} = 50$
- $p_{\text{C40}} = 229$
- $p_{\text{C41}} = 93$
- $p_{\text{C42}} = 169$
- $p_{\text{C43}} = 72$
- $p_{\text{C44}} = 67$
- $p_{\text{C45}} = 136$
- $p_{\text{C46}} = 118$
- $p_{\text{C47}} = 101$
- $p_{\text{C48}} = 94$
- $p_{\text{C49}} = 78$
- $p_{\text{C50}} = 76$
- $p_{\text{C51}} = 155$
- $p_{\text{C52}} = 114$
- $p_{\text{C53}} = 225$
- $p_{\text{C54}} = 238$
- $p_{\text{C55}} = 59$
- $p_{\text{C56}} = 135$
- $p_{\text{C57}} = 245$
- $p_{\text{C58}} = 231$
- $p_{\text{C59}} = 219$
- $p_{\text{C60}} = 167$
- $p_{\text{C61}} = 164$
- $p_{\text{C62}} = 139$
- $p_{\text{C63}} = 220$
- $p_{\text{C64}} = 167$
- $p_{\text{C65}} = 240$
- $p_{\text{C66}} = 170$
- $p_{\text{C67}} = 91$
- $p_{\text{C68}} = 106$
- $p_{\text{C69}} = 135$
- $p_{\text{C70}} = 91$
- $p_{\text{C71}} = 51$
- $p_{\text{C72}} = 73$
- $p_{\text{C73}} = 211$
- $p_{\text{C74}} = 189$
- $p_{\text{C75}} = 169$
- $p_{\text{C76}} = 153$
- $p_{\text{C77}} = 151$
- $p_{\text{C78}} = 167$
- $p_{\text{C79}} = 173$
- $p_{\text{C80}} = 152$
- $p_{\text{C81}} = 101$
- $p_{\text{C82}} = 216$
- $p_{\text{C83}} = 196$
- $p_{\text{C84}} = 92$
- $p_{\text{C85}} = 92$
- $p_{\text{C86}} = 97$
- $p_{\text{C87}} = 224$
- $p_{\text{C88}} = 128$
- $p_{\text{C89}} = 139$
- $p_{\text{C90}} = 109$
- $p_{\text{C91}} = 206$
- $p_{\text{C92}} = 161$
- $p_{\text{C93}} = 227$
- $p_{\text{C94}} = 187$
- $p_{\text{C95}} = 106$
- $p_{\text{C96}} = 248$
- $p_{\text{C97}} = 82$
- $p_{\text{C98}} = 222$
- $p_{\text{C99}} = 209$
- $p_{\text{C100}} = 223$
- $p_{\text{C101}} = 204$
- $p_{\text{C102}} = 114$
- $p_{\text{C103}} = 146$
- $p_{\text{C104}} = 231$
- $p_{\text{C105}} = 93$
- $p_{\text{C106}} = 224$
- $p_{\text{C107}} = 220$
- $p_{\text{C108}} = 100$
- $p_{\text{C109}} = 187$
- $p_{\text{C110}} = 213$
- $p_{\text{C111}} = 142$

**Unit processing times $a_{wi}$:**

For each workshop $w$ and component $i$, $a_{wi}$ is given by the corresponding value in processing_time_unit.csv. For example:
- For Casting: $a_{\text{Casting},\text{C1}} = 0.74$, $a_{\text{Casting},\text{C2}} = 0.77$, ..., $a_{\text{Casting},\text{C111}} = 3.81$
- For Milling: $a_{\text{Milling},\text{C1}} = 0.6$, $a_{\text{Milling},\text{C2}} = 3.38$, ..., $a_{\text{Milling},\text{C111}} = 2.67$
- For Finishing: $a_{\text{Finishing},\text{C1}} = 0.0$, $a_{\text{Finishing},\text{C2}} = 4.15$, ..., $a_{\text{Finishing},\text{C111}} = 4.07$
- For Assembly: $a_{\text{Assembly},\text{C1}} = 4.84$, $a_{\text{Assembly},\text{C2}} = 0.0$, ..., $a_{\text{Assembly},\text{C111}} = 4.02$
- For QA & Packaging: $a_{\text{QA \& Packaging},\text{C1}} = 0.92$, $a_{\text{QA \& Packaging},\text{C2}} = 0.0$, ..., $a_{\text{QA \& Packaging},\text{C111}} = 3.21$

All coefficients are as given in the retrieved data, in the original file and row order.

---

**Summary:**  
Choose integer $x_i \geq 0$ for each component $i$ to maximize total value, subject to the available working hours in each workshop, using the exact processing times and prices as above.