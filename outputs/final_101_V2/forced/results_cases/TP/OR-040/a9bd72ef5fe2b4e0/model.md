##### Decision Variables

Let $x_i$ be the integer number of development units chosen for area $i$ each day, for each area $i$ listed below.

##### Parameters

- Areas and benefit coefficients (in source order):

| Area                | Benefit Coefficient ($v_i$) | Unit Weight ($w_i$) |
|---------------------|-----------------------------|---------------------|
| Queens              | 443                         | 104                 |
| Brooklyn            | 522                         | 368                 |
| Manhattan           | 300                         | 483                 |
| Bronx               | 767                         | 165                 |
| Staten Island       | 300                         | 105                 |
| Harlem              | 309                         | 123                 |
| Upper East Side     | 598                         | 131                 |
| Lower Manhattan     | 460                         | 341                 |
| Midtown             | 318                         | 258                 |
| Long Island City    | 126                         | 469                 |
| Williamsburg        | 593                         | 387                 |
| Bushwick            | 871                         | 425                 |
| Flatbush            | 858                         | 482                 |
| Greenpoint          | 321                         | 495                 |
| Park Slope          | 275                         | 305                 |
| Astoria             | 700                         | 377                 |
| Jackson Heights     | 685                         | 318                 |
| Flushing            | 940                         | 56                  |
| Sunnyside           | 522                         | 213                 |
| Ditmars             | 763                         | 472                 |

- Overall development capacity: $C = 4466$

##### Mathematical Model

$\max \quad 443x_{\text{Queens}} + 522x_{\text{Brooklyn}} + 300x_{\text{Manhattan}} + 767x_{\text{Bronx}} + 300x_{\text{Staten Island}} + 309x_{\text{Harlem}} + 598x_{\text{Upper East Side}} + 460x_{\text{Lower Manhattan}} + 318x_{\text{Midtown}} + 126x_{\text{Long Island City}} + 593x_{\text{Williamsburg}} + 871x_{\text{Bushwick}} + 858x_{\text{Flatbush}} + 321x_{\text{Greenpoint}} + 275x_{\text{Park Slope}} + 700x_{\text{Astoria}} + 685x_{\text{Jackson Heights}} + 940x_{\text{Flushing}} + 522x_{\text{Sunnyside}} + 763x_{\text{Ditmars}}$

subject to

$104x_{\text{Queens}} + 368x_{\text{Brooklyn}} + 483x_{\text{Manhattan}} + 165x_{\text{Bronx}} + 105x_{\text{Staten Island}} + 123x_{\text{Harlem}} + 131x_{\text{Upper East Side}} + 341x_{\text{Lower Manhattan}} + 258x_{\text{Midtown}} + 469x_{\text{Long Island City}} + 387x_{\text{Williamsburg}} + 425x_{\text{Bushwick}} + 482x_{\text{Flatbush}} + 495x_{\text{Greenpoint}} + 305x_{\text{Park Slope}} + 377x_{\text{Astoria}} + 318x_{\text{Jackson Heights}} + 56x_{\text{Flushing}} + 213x_{\text{Sunnyside}} + 472x_{\text{Ditmars}} \leq 4466$

$x_i \in \mathbb{Z}_{\geq 0}$ for all areas $i$ listed above.