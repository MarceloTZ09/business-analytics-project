# Procedencia de los datos

- Autor: Frederik Fellenberg.
- Título: *Reward-Based Crowdfunding Data*.
- Repositorio: Mendeley Data, versión 1, publicada el 21 de abril de 2025.
- Fuente: https://data.mendeley.com/datasets/9r6v4hsc8x/1
- DOI: https://doi.org/10.17632/9r6v4hsc8x.1
- Licencia declarada: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
- Archivo utilizado: `data/cruda/crowdfunding_eu_data.csv`.
- Dimensiones verificadas: 2.272 campañas y 200 columnas.
- SHA-256: `dec820cebcb75c008cc1558879ee3a55c670c0439d467659dca61ac14d4002dc`.

La unidad es una campaña de Kickstarter originada en la UE. La fuente declara proyectos seleccionados aleatoriamente, lanzados desde el 5 de enero de 2022 y completados hasta el 8 de marzo de 2024. El archivo EU observado tiene inicios del 10 de enero de 2022 al 16 de febrero de 2024 y cierres del 7 de marzo de 2022 al 1 de marzo de 2024. La selección no constituye un censo de Kickstarter ni acredita representatividad de todo el crowdfunding europeo.

El CSV original se distribuye sin modificaciones. Las bases procesadas son adaptaciones: añaden duración calculada, dos proporciones corregidas, cuatro logaritmos, indicadores de auditoría, país derivado e indicadores de categoría. No se eliminan observaciones. Los diccionarios adaptan las definiciones de la fuente y explican las variables añadidas; no proceden de una nueva recolección.

Las 138 columnas de países describen residencia de patrocinadores, mientras que los 27 indicadores de origen identifican el país de la campaña. El archivo contiene campañas de 25 países de la UE; Chequia y Malta tienen indicadores constantes en cero. La suma geográfica de patrocinadores puede ser menor que el total; no se imputa esa diferencia.
