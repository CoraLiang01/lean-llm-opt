Let $x_k$ = number of units of component $k$ to produce, $k \in \{\text{C1}, \text{C2}, \ldots, \text{C111}\}$.

**Parameters:**

- $p_k$ = unit price of component $k$ (from unit_price.csv)
- $a_{wk}$ = unit processing time of component $k$ in workshop $w$ (from processing_time_unit.csv)
- $b_w$ = total available working hours in workshop $w$ (from total_working_hours.csv)

Workshops $w \in \{\text{Casting}, \text{Milling}, \text{Finishing}, \text{Assembly}, \text{QA \& Packaging}\}$.

---

**Mathematical Model:**

**Objective:**
\[
\max \sum_{k=1}^{111} p_k x_k
\]

**Subject to:**

_Casting capacity:_
\[
\sum_{k=1}^{111} a_{\text{Casting},k} \, x_k \leq 7650
\]

_Milling capacity:_
\[
\sum_{k=1}^{111} a_{\text{Milling},k} \, x_k \leq 6320
\]

_Finishing capacity:_
\[
\sum_{k=1}^{111} a_{\text{Finishing},k} \, x_k \leq 5538
\]

_Assembly capacity:_
\[
\sum_{k=1}^{111} a_{\text{Assembly},k} \, x_k \leq 5957
\]

_QA & Packaging capacity:_
\[
\sum_{k=1}^{111} a_{\text{QA \& Packaging},k} \, x_k \leq 6988
\]

_Nonnegativity and integrality:_
\[
x_k \in \mathbb{Z}_{\geq 0} \qquad \forall k \in \{\text{C1}, \ldots, \text{C111}\}
\]

---

**Where:**

- $p_k$ is as follows (from unit_price.csv, in source order):

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

- $a_{w,k}$ is the value in row $w$ and column $k$ of processing_time_unit.csv, in the original file order.

- $b_{\text{Casting}} = 7650$
- $b_{\text{Milling}} = 6320$
- $b_{\text{Finishing}} = 5538$
- $b_{\text{Assembly}} = 5957$
- $b_{\text{QA \& Packaging}} = 6988$

All coefficients and indices are as above, and all variables $x_k$ are nonnegative integers.