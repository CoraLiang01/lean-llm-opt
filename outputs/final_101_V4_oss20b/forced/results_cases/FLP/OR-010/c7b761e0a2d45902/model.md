##### Decision Variables

$x_i \in \mathbb{Z}_{\ge 0}$: quantity of product $i$ fulfilled, for $i \in I$.

##### Objective Function

$\displaystyle \max \sum_{i\in I} r_i\,x_i$

##### Constraints

1. Demand satisfaction: $\displaystyle x_i \le d_i,\quad \forall i\in I$  
2. Inventory limitation: $\displaystyle x_i \le I_i,\quad \forall i\in I$  
3. Domain: $x_i \in \mathbb{Z}_{\ge 0},\quad \forall i\in I$

$I=\{1,2,\dots,71\}$

##### Retrieved Information

**Product list**  
$P=\bigl[\text{Adams Group_service},\text{Anderson-Leach_against},\text{Anderson-Valdez_somebody},\text{Anderson-White_son},\text{Andrews LLC_matter},\text{Andrews PLC_enter},\text{Andrews-Martin_build},\text{Arias-Mendoza_life},\text{Bennett and Sons_down},\text{Bennett, Foster and Moreno_enter},\text{Fernandez, Long and Nelson_member},\text{Fernandez-Fischer_million},\text{Ferrell Inc_prove},\text{Fields, Christensen and Daniels_nor},\text{Figueroa LLC_involve},\text{Finley Group_itself},\text{Fisher Ltd_speak},\text{Fisher-Marshall_cover},\text{Fuller-Torres_behavior},\text{Fuller-Walters_radio},\text{Gallagher-Campbell_dark},\text{Gallagher-Kirby_talk},\text{Gallagher-Parker_recognize},\text{Gonzalez PLC_task},\text{Gonzalez, Coleman and Le_heavy},\text{Gonzalez, Lowe and Robinson_education},\text{Gonzalez-Horn_light},\text{Good, Davis and Smith_station},\text{Goodman, Hughes and White_realize},\text{Goodwin PLC_bad},\text{Gordon PLC_detail},\text{Graham LLC_stage},\text{Graham Ltd_marriage},\text{Graham-Swanson_message},\text{Grant, Cross and Bennett_religious},\text{Grant, Mcdonald and Watson_owner},\text{Graves, Turner and Crawford_wait},\text{Gray, Smith and Barnes_much},\text{Green Inc_direction},\text{Green-Rogers_could},\text{Greene-Baxter_them},\text{Greer and Sons_keep},\text{Griffin, Boyle and Dawson_anything},\text{Harper Inc_well},\text{Harper Ltd_off},\text{Harrington, Sosa and Mccarty_coach},\text{Harris Group_different},\text{Harris and Sons_audience},\text{Harris and Sons_fear},\text{Harris, Hamilton and Rose_contain},\text{Harris, Stevens and Hall_answer},\text{Harris-Bell_painting},\text{Harris-Melton_often},\text{Harris-Rogers_after},\text{Hart Group_cold},\text{Hutchinson, Roberts and Mcbride_among},\text{Ibarra, Jackson and Potter_upon},\text{Jackson Inc_anyone},\text{Jackson LLC_report},\text{Jackson, Collier and Barber_result},\text{Jackson, White and Brown_expect},\text{Jackson-Carroll_six},\text{Lee-Jones_soldier},\text{Lewis, Sanchez and Turner_point},\text{Lin LLC_performance},\text{Lindsey, Avila and Brown_candidate},\text{Lindsey-Strickland_modern},\text{Little-Perez_choice},\text{Lloyd, Stone and Mcguire_national},\text{Long, Hughes and Gallegos_receive},\text{Lopez PLC_analysis}\bigr]$

**Revenue vector**  
$r=\bigl[1003.56,\;549.92,\;169.7,\;1285.81,\;1269.71,\;805.82,\;1134.19,\;1479.23,\;139.14,\;518.67,\;1088.57,\;371.56,\;956.78,\;1283.7,\;1184.54,\;377.8,\;701.97,\;875.91,\;951.85,\;869.95,\;887.06,\;1318.33,\;448.61,\;167.99,\;1068.1,\;872.16,\;483.26,\;1279.24,\;894.34,\;367.91,\;178.22,\;1030.8,\;479.44,\;973.62,\;1211.35,\;467.93,\;678.72,\;478.36,\;1196.95,\;1118.63,\;972.1,\;338.6,\;178.52,\;992.14,\;498.77,\;965.34,\;321.11,\;287.36,\;462.26,\;641.22,\;1136.87,\;1385.88,\;867.3,\;1175.92,\;153.61,\;753.55,\;245.81,\;507.13,\;513.64,\;1316.82,\;1409.21,\;1237.37,\;1042.66,\;156.02,\;728.62,\;971.93,\;1128.07,\;256.42,\;1064.72,\;348.43,\;1183.94\bigr]$

**Demand vector**  
$d=\bigl[10,\;6,\;75,\;121,\;130,\;112,\;125,\;54,\;75,\;102,\;48,\;4,\;125,\;9,\;132,\;29,\;43,\;5,\;40,\;48,\;87,\;79,\;20,\;33,\;60,\;25,\;42,\;52,\;116,\;95,\;66,\;25,\;78,\;138,\;32,\;28,\;84,\;102,\;112,\;7,\;62,\;109,\;81,\;131,\;26,\;109,\;65,\;121,\;90,\;31,\;24,\;122,\;101,\;76,\;100,\;125,\;45,\;80,\;64,\;98,\;102,\;40,\;110,\;59,\;97,\;90,\;43,\;96,\;21,\;55,\;30\bigr]$

**Initial inventory vector**  
$I=\bigl[80,\;40,\;560,\;820,\;960,\;850,\;950,\;360,\;580,\;800,\;340,\;30,\;990,\;70,\;890,\;240,\;290,\;40,\;300,\;350,\;600,\;610,\;160,\;260,\;480,\;200,\;290,\;350,\;900,\;760,\;440,\;170,\;600,\;920,\;240,\;190,\;600,\;760,\;850,\;50,\;470,\;790,\;580,\;880,\;170,\;790,\;500,\;890,\;680,\;250,\;170,\;830,\;680,\;590,\;700,\;830,\;360,\;570,\;510,\;690,\;760,\;270,\;880,\;470,\;670,\;620,\;310,\;670,\;150,\;440,\;210\bigr]$