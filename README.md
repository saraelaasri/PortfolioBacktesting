# PortfolioBacktesting
Portfolio opimisation with the optimisation models ( Modèle de Black-Litterman , Modèle Mean-Variance , L’algorithme génétique )
## Interface Web
Pour fournir aux utilisateurs une plateforme dynamique et efficace pour gérer leurs
investissements. Il leur permet de calculer les différents poids de leurs portefeuilles en
utilisant trois modèles différents, avec la possibilité de paramétrer ces modèles selon leurs
propres préférences.
Les technologies utilisées sont :
• Python : pour la mise en œuvre des modèles d’optimisation et des calculs associés.
• Flask : Nous avons utilisé Flask comme framework de développement web pour le
côté backend de notre application. Flask est un framework léger et flexible qui nous
permettra de créer des API robustes pour gérer les requêtes et les interactions avec
les modèles d’optimisation. Nous pourrons ainsi traiter les paramètres entrés par
les utilisateurs, effectuer les calculs nécessaires et renvoyer les résultats de manière
efficace.
• HTML, CSS, JS : Pour la partie front-end de notre application, nous avons utilisé
les technologies web standards telles que HTML, CSS . HTML sera utilisé pour la
structure de la page, CSS pour la mise en forme et le style.
Dans l’interface, nous avons divisé le processus en quatre étapes principales. Voici un
aperçu de chacune de ces étapes :
### Étape 1 : les données d’entrée
Cette première étape est dédiée à la saisie des informations nécessaires pour effectuer
les calculs.
![image](https://github.com/saraelaasri/PortfolioBacktesting/assets/91394848/d3220c49-9769-487f-bd1d-12be9cfb0238)
![image](https://github.com/saraelaasri/PortfolioBacktesting/assets/91394848/9f1da2a8-8477-47b0-89c1-a93c6d800bfc)
• Sélection des cinq actifs : L’interface présentera cinq zones de saisie où les utilisateurs pourront entrer les symboles des titres des actifs qu’ils souhaitent sélectionner.
• Saisie des vues : Pour le modèle Black Litterman les utilisateurs saisiront leurs
vues dans les zones correspondantes en fonction de leurs anticipations personnelles
pour chaque actif. Les vues sont sous la forme de rendements attendus pour chaque
actif.
• Paramètres de l’algorithme génétique :
– Population : Choisir le nombre d’individus dans la population qui évoluera
au fil des générations.
– Génération : Les utilisateurs peuvent spécifier le nombre de générations (itérations) pour lesquelles l’algorithme génétique sera exécuté.
– Mutation : les utilisateurs ont la possibilité de spécifier les mutations. Les
mutations sont des modifications aléatoires appliquées aux individus de la population afin d’introduire de la diversité génétique et d’explorer de nouvelles
solutions potentielles.
– Élitisme : les utilisateurs peuvent choisir la proportion des meilleurs individus
à préserver de chaque génération.
– Risk free rate : les utilisateurs peuvent saisir le taux sans risque en fonction
des conditions économiques et de leurs préférences.
• Nombre de portefeuilles : Les utilisateurs peuvent spécifier le nombre de portefeuilles qu’ils souhaitent générer dans l’interface. Ce nombre détermine la diversité
des portefeuilles qui seront générés et évalués par le modèle mean-variance.
• Date début : la date à partir de laquelle les données historiques des actifs financiers
seront prises en compte dans l’optimisation de portefeuille.
• Date fin : correspond à la date à laquelle les données historiques des actifs financiers
se terminent pour l’optimisation du portefeuille.
###  Étape 2 : statistique et choix de portefeuille
Dans l’étape statistique et choix de portefeuille, l’interface présente les graphiques
des prix et des statistiques effectués sur les données fournies. Cette section permet aux
utilisateurs de visualiser et d’analyser les performances passées des actifs, afin de prendre
des décisions aux actifs à inclure dans leur portefeuille.
![image](https://github.com/saraelaasri/PortfolioBacktesting/assets/91394848/b01aac9d-b31c-4424-90a3-5c2d6a48bec4)
![image](https://github.com/saraelaasri/PortfolioBacktesting/assets/91394848/6e57e91b-177b-4f48-959e-9654144a8925)
Cette étape affiche le graphe et les statistiques liées aux données fournies :
• Graphiques des prix : l’interface affiche des graphiques des prix historiques des
actifs sélectionnés. Ces graphiques permettent aux utilisateurs de visualiser les tendances et les variations des séries des prix au fil du temps et ainsi leur permettre
d’évaluer la volatilité des actifs et identifier les périodes qui ont influencé les performances.
• Statistiques des performances : l’interface fournit des statistiques des performances basées sur les données des rendements des prix historiques des actifs.
• Matrice de corrélation : la matrice de corrélation permet aux utilisateurs d’interpréter les relations entre les actifs. Une corrélation positive élevée entre deux
actifs suggère qu’ils se déplacent dans la même direction, tandis qu’une corrélation
négative élevée indique qu’ils se déplacent dans des directions opposées. Les utilisateurs peuvent utiliser ces informations pour diversifier et bien choisir les actifs de
leur portefeuille.
### Étape 3 : Allocation et Optimisation
Dans l’étape d’allocation et d’optimisation, l’interface permet aux utilisateurs de définir les poids de chaque actif dans leur portefeuille en utilisant différentes méthodes
d’optimisation. Une fois que les poids sont définis, l’interface calcule les ratios et les mesures de performance pour évaluer et comparer les différentes allocations.
![image](https://github.com/saraelaasri/PortfolioBacktesting/assets/91394848/157bb943-90a6-4c15-a43c-848aed7d503f)
![image](https://github.com/saraelaasri/PortfolioBacktesting/assets/91394848/22ed6a73-6732-49e9-aceb-921a005b5ac0)
Pour faciliter la comparaison des allocations de portefeuille basées sur différentes mé-
thodes d’optimisation, plusieurs éléments sont inclus :
• Figure des portefeuilles aléatoires (Mean-Variance) : la figure représentant
les portefeuilles aléatoires générés à partir de la méthode Mean-Variance. Cette
figure montre généralement la volatilité en fonction du rendement pour les diffé-
rentes allocations de portefeuille. Les portefeuilles aléatoires sont représentés par
des points, formant une courbe de la ”frontière efficiente”.
• Tableau des poids des trois modèles : le tableau affiche les poids calculés pour
chaque actif dans le portefeuille par les trois modèles d’optimisation. Ce tableau
permet aux utilisateurs de visualiser rapidement la répartition des poids pour chaque
modèle.
• Figure des poids : la figure graphique représente visuellement les poids attribués
à chaque actif dans les différents portefeuilles, permettant aux utilisateurs de voir
rapidement les différences de répartition des poids entre les modèles.
• Différents ratios de performance : ces ratios permettent aux utilisateurs de
comparer les performances des différents modèles et de prendre des décisions sur la
base de ces mesures.
### Étape 4 : Backtesting
Dans l’étape de backtesting, l’interface permet aux utilisateurs de tester les perfor-
mances passées des différentes modèles d’optimisation. Cela leur permet d’évaluer la per-
formance de chaque modèle dans des conditions historiques et de prendre des décisions
éclairées sur le choix du modèle le plus approprié pour leurs stratégies d’investissement .
![image](https://github.com/saraelaasri/PortfolioBacktesting/assets/91394848/8c58b7bb-17b7-4330-b6f3-3365eb5d1dae)
![image](https://github.com/saraelaasri/PortfolioBacktesting/assets/91394848/a2848f26-04ea-44bb-b87a-92d14d0f8bbd)
La dernière section affiche les éléments suivants :
• Graphique des rendements cumulés du portefeuille : L’interface affiche un
graphique montrant l’évolution des rendements cumulés du portefeuille basé sur le
modèle pour la période de backtesting et les rendements cumulés de chaque actif
individuel inclus dans le portefeuille. Cela permet aux utilisateurs de voir comment
le portefeuille a performé dans le temps en utilisant le modèle choisi, et de comparer
la performance du portefeuille par rapport à celle des actifs et de voir si le modèle
a réussi à sur performer certains actifs ou non.
• Un tableau récapitulatif créé pour fournir aux utilisateurs un aperçu des perfor-
mances des différents modèles d’optimisation contenant le profit net total, le ratio
de Sharpe, le nombre total de transactions et le Maximum Drawdown.










