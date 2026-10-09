Let x_i be the integer number of development units per day in area i, for each area i listed below.

Indices and coefficients (in original file order):

| Area                | Benefit Coefficient (Value) |
|---------------------|----------------------------|
| Queens              | 443                        |
| Brooklyn            | 522                        |
| Manhattan           | 300                        |
| Bronx               | 767                        |
| Staten Island       | 300                        |
| Harlem              | 309                        |
| Upper East Side     | 598                        |
| Lower Manhattan     | 460                        |
| Midtown             | 318                        |
| Long Island City    | 126                        |
| Williamsburg        | 593                        |
| Bushwick            | 871                        |
| Flatbush            | 858                        |
| Greenpoint          | 321                        |
| Park Slope          | 275                        |
| Astoria             | 700                        |
| Jackson Heights     | 685                        |
| Flushing            | 940                        |
| Sunnyside           | 522                        |
| Ditmars             | 763                        |

Let x_Queens, x_Brooklyn, ..., x_Ditmars ∈ {0, 1, 2, ...} (all integer and nonnegative).

Model:

Maximize
 443·x_Queens + 522·x_Brooklyn + 300·x_Manhattan + 767·x_Bronx + 300·x_StatenIsland + 309·x_Harlem + 598·x_UpperEastSide + 460·x_LowerManhattan + 318·x_Midtown + 126·x_LongIslandCity + 593·x_Williamsburg + 871·x_Bushwick + 858·x_Flatbush + 321·x_Greenpoint + 275·x_ParkSlope + 700·x_Astoria + 685·x_JacksonHeights + 940·x_Flushing + 522·x_Sunnyside + 763·x_Ditmars

Subject to
 x_Queens + x_Brooklyn + x_Manhattan + x_Bronx + x_StatenIsland + x_Harlem + x_UpperEastSide + x_LowerManhattan + x_Midtown + x_LongIslandCity + x_Williamsburg + x_Bushwick + x_Flatbush + x_Greenpoint + x_ParkSlope + x_Astoria + x_JacksonHeights + x_Flushing + x_Sunnyside + x_Ditmars ≤ 4466

 x_i ∈ {0, 1, 2, ...} for all areas i listed above.

This model maximizes total benefit from daily development across all areas, subject to the overall development capacity.