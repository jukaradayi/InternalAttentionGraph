Instructions pour Utiliser Gephi
================================

I Charger le graphe
-------------------

* Charger le graphe se fait en deux temps: 

    * D'abord charger le graphe commonCitationGraph.csv, cliquer sur "suivant" puis "terminer" sur la première fenêtre "Spreadsheet. Ensuite un "rapport d'importation"  s'ouvre, choisir "non dirigé" ou "undirected" dans le menu déroulant "type de graphe", puis "okay".

    * Ensuite, il faut importer la detection de communauté, cliquer sur "fichier", "ouvrir", puis choisir le fichier "communities_res[...].csv" avec la résolution désirée. Cliquer sur "suivant" puis "terminer" sur la première fenêtre "Spreadsheet. Dans le "rapport d'importation", choisir "non dirigé", et **"ajouter à l'espace de travail existant"**.

II Manipuler le graphe
----------------------

* Pour la mise en forme et les métriques, il faut être dans la section "overview" que tu devrais pouvoir selectionner en haut.

* Pour modifier la mise en forme il faut choisir des options dans "layout". Certaines options de layout ne font que modifier le layout en cours (contraction, expansion, noverlap, rotate..). Pour celle que je t'ai montré, j'ai choisi "Yifan Hu Proportional" puis "noverlap" - noverlap tourne indéfiniment parfois, tu peux le "stop" au bout de quelques secondes (noverlap évite que les noeuds ne se chevauchent).


* Tu peux visualiser les partitionnement en communauté en allant dans "aspect" en haut a gauche : nodes -> partition -> Community -> apply.

* Pour chaque tu peux choisir si tu le représente via la couleur ou la taille des noeuds en cliquant sur les icones à droite de "Nodes Edges" .

