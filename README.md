# **Rapport de Projet — Atlantic Haven Hotels**

## **Examen Final Machine Learning & Data Science — M1**

Réalisé au sein de **ISPM — Madagascar** ([www.ispm-edu.com](https://www.ispm-edu.com))

---

### **1. Informations sur le Groupe**

> ⚠️ **À COMPLÉTER** — je n'ai pas les identités des membres de l'équipe. Remplis chaque bloc ci-dessous.

#### Membre 1

- nom :Randriantseheno
- prénom(s) :Mitandro Ny Aina Arivelo
- classe :isaia4
- numéro :24
- rôle : inscription

#### Membre 2

- nom :Randriamahefa 
- prénom(s) :Tsilavina Mia
- classe :isaia4
- numéro :23
- rôle : présentateur

#### Membre 3

- nom :Ratovomanalina
- prénom(s) :Sitraka Mamy
- classe :isaia4
- numéro :19
- rôle : développeur

#### Membre 4

- nom :Ramalarison 
- prénom(s) :Tsiory Nomena
- classe :isaia4
- numéro :22
- rôle : responsable de la modélisation

#### Membre 5

- nom :Randrianaliarimanana
- prénom(s) :Manoasoa
- classe :isaia4
- numéro :21
- rôle : analyste
---

### **2. Résumé du Travail**

#### Problématique

Atlantic Haven Hotels exploite des établissements dans dix régions italiennes couvrant des contextes touristiques très variés (urbain, balnéaire, montagnard, insulaire). Une annulation tardive laisse une chambre inoccupée sans possibilité de la revendre à temps, ce qui perturbe la planification opérationnelle et représente une perte de revenu. L'enjeu est donc de détecter, le plus tôt possible après la réservation, les dossiers présentant un risque élevé d'annulation, afin de permettre une action commerciale ciblée — sans pour autant pénaliser les clients qui, en réalité, maintiendront leur séjour.

#### Méthodologie adoptée

1. **EDA et nettoyage** : traitement des valeurs manquantes (`agent_id` → indicateur `is_direct_booking` + catégorie `DIRECT` ; `prix_moyen_nuit_eur` et `enfants` → médiane ; `demandes_speciales` et `marche_origine` → valeurs par défaut explicites), correction d'un mélange de types sur `demandes_speciales`, vérification de la redondance entre `region_hotel`, `ville` et `type_destination`.
2. **Validation temporelle** : les données sont triées par `date_arrivee` puis découpées 80/20 (train = réservations les plus anciennes, validation = les 20 % les plus récentes), afin de reproduire la situation réelle du jeu de test (composé de réservations postérieures au train).
3. **Baseline** : régression logistique sur pipeline `ColumnTransformer` (imputation + `StandardScaler` pour le numérique, imputation + `OneHotEncoder` pour le catégoriel), appris uniquement sur le train.
4. **Comparaison de modèles** : Random Forest et Gradient Boosting comparés à la baseline sur le même split temporel ; réglage d'hyperparamètres du Random Forest via `GridSearchCV` avec `TimeSeriesSplit` (5 découpages).
5. **Correction du déséquilibre de classes** (~72 % maintenues / ~28 % annulées) : entraînement des 3 familles de modèles avec `sample_weight` équilibré (`compute_sample_weight`), puis recherche du seuil de décision optimal (maximisant le F1) propre à chaque modèle plutôt que le seuil par défaut de 0.5.
6. **Feature engineering** : ajout de variables temporelles (délai réservation→arrivée, mois/jour de semaine d'arrivée, week-end), liées au séjour (nuits/personnes par chambre), au prix (prix par personne, remise appliquée) et à l'historique client (taux d'annulation passée, client nouveau) — réentraînement du modèle gagnant et comparaison du F1 avant/après.
7. **Interprétation** : importance des variables du modèle final, analyse des faux positifs / faux négatifs sur la validation.

#### Résultats obtenus

Le meilleur F1-score sur la validation temporelle est de **0.5000**, obtenu avec un **Gradient Boosting entraîné avec `sample_weight` équilibré, au seuil de décision 0.45** (contre un F1 de seulement 0.10 pour le Random Forest et 0.4850 pour le Gradient Boosting au seuil par défaut 0.5 — le rééquilibrage et l'ajustement du seuil étaient donc indispensables).

Découverte importante : le feature engineering testé (variables temporelles, de séjour, de prix et d'historique) **n'a pas apporté de gain net** — F1 = 0.4924 avec les nouvelles variables contre 0.5000 sans. Le modèle final retenu est donc le Gradient Boosting balanced **sans** ces variables ajoutées, ce qui illustre qu'une augmentation de la complexité n'est pas toujours bénéfique.

#### Mots-clés

Classification binaire, annulation de réservation, déséquilibre de classes, validation temporelle, `sample_weight`, F1-score, seuil de décision, Gradient Boosting.

---

### **3. Contenu du Repository**

- **notebook.ipynb** : code complet de l'EDA, du prétraitement, de la modélisation et de l'évaluation ;
- **submission.csv** : prédictions sur `reservations_test.csv` ;
- **README.md** : présent rapport ;
- **requirements.txt** : *(à ajouter si nécessaire — voir section 7)*.

**🔗 Liens utiles :**

- [**LIEN VERS LA VIDÉO DE PRÉSENTATION**] → ⚠️ **À COMPLÉTER**

---

### **4. Résultats de Modélisation**

Résultats sur la validation temporelle (20 % des réservations les plus récentes du train), chaque modèle évalué à **son propre seuil optimal** (comparaison la plus juste pour un problème déséquilibré) :

| Modèle | Paramètres principaux | Seuil retenu | F1-score | Précision | Rappel | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| Régression logistique — baseline (non rééquilibrée) | `max_iter=1000` | 0.50 | ⚠️ non relevé dans nos échanges — à recopier depuis la cellule 2.4 du notebook | — | — | — |
| Régression logistique (balanced) | `sample_weight` équilibré | 0.39 | 0.4882 | 0.3438 | 0.8420 | 0.6775 |
| Random Forest (balanced) | `n_estimators=300`, `sample_weight` équilibré | 0.23 | 0.4844 | 0.3575 | 0.7449 | 0.6491 |
| **Gradient Boosting (balanced) — modèle final** | `sample_weight` équilibré, sans feature engineering | **0.45** | **0.5000** | **0.3692** | **0.7743** | **0.6746** |

> Précision / Rappel / ROC-AUC calculés et vérifiés pour les 3 modèles (ex. Gradient Boosting : 2×0.3692×0.7743/(0.3692+0.7743) = 0.5000, cohérent avec le F1 rapporté).

**Seuil de décision retenu :** 0.45 (Gradient Boosting balanced), contre 0.5 par défaut.

**Justification du choix du modèle final :**

Le Gradient Boosting balanced obtient le meilleur F1 des trois familles testées (0.5000, contre 0.4882 pour la régression logistique et 0.4844 pour le Random Forest, chacun à son seuil optimal). Le détail des métriques (précision 0.3692, rappel 0.7743, ROC-AUC 0.6746) confirme le choix fait sur le seuil : le modèle privilégie fortement le rappel — il détecte plus de 3 annulations réelles sur 4 — au prix d'une précision modeste (un peu plus d'1 alerte sur 3 est une vraie annulation). C'est cohérent avec la mission du sujet : mieux vaut multiplier les relances (faux positifs, peu coûteux si l'action reste proportionnée — voir Q7) que manquer des annulations réelles (faux négatifs, coût opérationnel direct). Le ROC-AUC de 0.67 indique un pouvoir discriminant modéré mais réel, cohérent avec la difficulté intrinsèque du problème — les trois modèles balanced obtiennent d'ailleurs un ROC-AUC très proche (0.6775 pour la régression logistique, 0.6491 pour le Random Forest, 0.6746 pour le Gradient Boosting), ce qui suggère que la limite vient surtout de l'information disponible dans les données plutôt que du choix d'algorithme. À noter que la régression logistique atteint un rappel encore supérieur (0.8420 contre 0.7743) mais avec une précision légèrement plus faible (0.3438 contre 0.3692) : le Gradient Boosting reste préféré car il maximise le F1, mais la régression logistique — plus simple et plus interprétable — resterait une alternative défendable si l'hôtel privilégiait explicitement le rappel au détriment de la précision. Le feature engineering testé n'ayant pas amélioré le score, la version la plus simple (sans les variables ajoutées) a été conservée, ce qui limite aussi le risque de sur-apprentissage et facilite l'interprétation.

---

### **5. Réponses aux Questions d'Analyse**

#### **Q1. Pourquoi utilise-t-on principalement le F1-score plutôt que l'accuracy pour cette tâche ?**

La cible est déséquilibrée (~72 % de réservations maintenues contre ~28 % d'annulations). Un modèle qui prédirait systématiquement « maintenue » atteindrait ~72 % d'accuracy sans détecter une seule annulation — ce chiffre serait donc trompeur. Le F1-score, moyenne harmonique de la précision et du rappel calculée sur la classe minoritaire (« annulée »), pénalise à la fois les annulations manquées (faux négatifs) et les fausses alertes (faux positifs), ce qui correspond directement à la mission du sujet : détecter les vraies annulations sans pénaliser inutilement les clients qui maintiennent leur séjour.

#### **Q2. Dans ce contexte, qu'est-ce qui est le plus grave : un faux positif ou un faux négatif ?**

- **Faux positif** : un client qui va en réalité maintenir son séjour, mais classé « à risque » — il pourrait être pénalisé à tort (demande de prépaiement, sur-booking de sa chambre, communication intrusive).
- **Faux négatif** : une annulation réelle non détectée — la chambre reste bloquée dans la planification jusqu'à l'annulation effective, sans anticipation possible.

Réponse nuancée : le sujet insiste explicitement sur le fait de « ne pas pénaliser inutilement les clients susceptibles de maintenir leur séjour », ce qui pousserait à limiter les faux positifs. Mais un faux négatif a un coût opérationnel direct (chambre non revendue à temps) alors qu'un faux positif, si l'action déclenchée reste proportionnée (voir Q7 — une simple relance, pas une annulation forcée), a un coût plus faible. Le choix du seuil 0.45 (plutôt que plus haut) privilégie donc le rappel : sur la validation, on préfère avoir 653 faux positifs (relances superflues mais peu coûteuses) plutôt que de laisser passer davantage d'annulations réelles.

#### **Q3. Quelles variables créées par feature engineering ont le plus amélioré votre modèle par rapport à la régression logistique de référence ?**

Aucune. Les variables testées — `delai_reservation_jours`, `mois_arrivee`, `jour_semaine_arrivee`, `arrivee_weekend`, `nuits_par_chambre`, `personnes_par_chambre`, `prix_total_par_personne`, `remise_appliquee`, `taux_annulation_passee`, `client_nouveau` — ont fait **baisser** le F1 du modèle gagnant (Gradient Boosting balanced) de 0.5000 à 0.4924. Ce résultat négatif est documenté plutôt que masqué : il suggère que l'information utile est déjà largement capturée par les variables brutes les plus importantes (`tarif_remboursable`, `type_acompte`, `prix_moyen_nuit_eur`, `montant_total_eur`), et que les nouvelles variables ont surtout ajouté du bruit ou de la colinéarité sans apporter de signal supplémentaire exploitable par le modèle.

#### **Q4. Pourquoi un découpage aléatoire simple peut-il produire une évaluation trompeuse sur ce dataset ?**

Un découpage aléatoire mélangerait des réservations passées et futures dans le train et dans la validation, ce qui permettrait au modèle d'être indirectement évalué sur des données antérieures à certaines de ses données d'entraînement — une forme de fuite temporelle. Or `reservations_test.csv` est explicitement composé de réservations **plus récentes** que le train. Notre validation reproduit cette contrainte : `df` est trié par `date_arrivee`, puis découpé strictement en 80 % (train, réservations les plus anciennes) / 20 % (validation, réservations les plus récentes) via `split_index = int(len(df) * 0.8)`. Le réglage d'hyperparamètres (`GridSearchCV`) utilise également un `TimeSeriesSplit` à 5 découpages plutôt qu'une validation croisée classique, pour la même raison.

#### **Q5. Quels profils ou scénarios de réservation sont les plus fréquemment associés aux annulations dans vos analyses ?**

D'après l'importance des variables du modèle final (section 5.1 du notebook), les scénarios les plus associés au risque d'annulation prédit sont :

- **Le type de tarif et d'acompte** : `tarif_remboursable` et `type_acompte` sont les deux variables les plus importantes du modèle — un tarif remboursable et l'absence d'acompte (ou un acompte partiel) réduisent l'engagement du client et sont associés à un risque plus élevé, à l'inverse d'un acompte total.
- **Un délai long entre réservation et arrivée** (`delai_reservation_jours`, 3ᵉ variable la plus importante) : plus ce délai est long, plus le client a d'opportunités de changer d'avis.
- **Un montant ou un prix par nuit élevé** (`prix_moyen_nuit_eur`, `montant_total_eur`, `prix_total_par_personne`) : les réservations les plus coûteuses semblent plus sensibles à l'annulation.

> ⚠️ Ces profils sont déduits du classement d'importance des variables, pas d'un croisement chiffré direct (ex. « taux d'annulation par tranche de délai »). Si le canevas exige des chiffres précis par profil, il faut ajouter un `groupby('reservation_annulee')` sur ces variables — dis-le-moi et je génère le code.
>
> Conformément à la consigne du canevas : ces profils décrivent des **circonstances de réservation** (conditions commerciales, délai, prix), pas une région ou une population — aucune région italienne n'est présentée comme intrinsèquement à risque.

#### **Q6. Comment votre pipeline traite-t-il les valeurs manquantes et les catégories jamais observées pendant l'entraînement ?**

- **Valeurs manquantes** : le `ColumnTransformer` applique un `SimpleImputer(strategy='median')` sur les variables numériques et un `SimpleImputer(strategy='most_frequent')` sur les variables catégorielles, **appris uniquement sur `X_train`** puis réappliqué tel quel sur `X_val`/`X_test` via `.transform()` — aucune statistique n'est recalculée sur la validation ou le test, ce qui évite toute fuite de données.
- **Catégories jamais vues** : le `OneHotEncoder(handle_unknown='ignore')` encode toute catégorie absente du train en un vecteur de zéros plutôt que de lever une erreur, ce qui permet au pipeline de rester utilisable sur des données futures sans réentraînement.
- Le split train/validation est strictement temporel (Q4), et les traitements de nettoyage appliqués au jeu de test (médianes, valeurs par défaut) réutilisent explicitement les statistiques apprises sur le train (`df`), jamais recalculées sur le test.

#### **Q7. Selon vous, quelle action l'hôtel devrait-il entreprendre lorsqu'une réservation en cours présente une forte probabilité d'annulation ?**

Utiliser `probabilite_annulation` comme un signal graduel plutôt que la seule décision binaire : par exemple, une relance automatique (email de confirmation, rappel des conditions) pour les probabilités modérées, et une action commerciale plus ciblée (proposition d'un acompte partiel, offre de flexibilité) uniquement au-delà du seuil retenu (0.45) — jamais une annulation ou un sur-booking automatique de la chambre, ce qui pénaliserait à tort les nombreux faux positifs identifiés en section 9.

#### **Q8. Votre modèle présente-t-il des performances comparables selon les régions ou les types de destination ?**

Le F1 varie sensiblement selon la région, de **0.4110 (Campanie)** à **0.6000 (Sicile)** — un écart de près de 0.19, non négligeable :

| Région | n (validation) | F1-score |
|---|---:|---:|
| Sicilia | 128 | 0.6000 |
| Trentino-Alto Adige | 155 | 0.5405 |
| Lombardia | 216 | 0.5275 |
| Puglia | 98 | 0.5195 |
| Sardegna | 76 | 0.5152 |
| Lazio | 253 | 0.5096 |
| Veneto | 206 | 0.4606 |
| Toscana | 181 | 0.4595 |
| Liguria | 117 | 0.4565 |
| Campania | 170 | 0.4110 |

**Limites à considérer** : les régions les plus faibles en effectif (Sardegna, n=76 ; Puglia, n=98) donnent un F1 statistiquement moins fiable — une poignée d'erreurs en plus ou en moins y fait bouger le score de façon disproportionnée, contrairement aux régions bien représentées comme Lazio (n=253) ou Lombardia (n=216). L'écart entre Campania (0.4110) et Sicilia (0.6000) mérite d'être surveillé, mais avec seulement 170 et 128 observations respectivement, il n'est pas certain qu'il se maintienne sur un échantillon plus large — une conclusion définitive sur une hétérogénéité géographique demanderait davantage de données de validation par région.

#### **Q9. Analyse des erreurs**

Sur la validation (modèle final, seuil 0.45) :

**5 faux positifs** (le client maintient en réalité, mais le modèle prédit une annulation) :

| reservation_id | région | canal | acompte | tarif remboursable | délai (jours) | proba |
|---|---|---|---|---|---:|---:|
| R002768 | Lombardia | agence | partiel | non | 526 | 0.7979 |
| R000847 | Campania | telephone | aucun | oui | 25 | 0.5528 |
| R004439 | Campania | plateforme_en_ligne | partiel | oui | 50 | 0.5379 |
| R006375 | Toscana | plateforme_en_ligne | aucun | oui | 24 | 0.5676 |
| R003715 | Lazio | site_hotel | aucun | oui | 12 | 0.5080 |

**5 faux négatifs** (le client annule en réalité, mais le modèle prédit un maintien) :

| reservation_id | région | canal | acompte | tarif remboursable | délai (jours) | proba |
|---|---|---|---|---|---:|---:|
| R004499 | Campania | site_hotel | partiel | non | 10 | 0.4395 |
| R008488 | Toscana | site_hotel | total | non | 37 | 0.0908 |
| R005823 | Veneto | telephone | partiel | oui | 54 | 0.4134 |
| R002923 | Sardegna | plateforme_en_ligne | total | non | 29 | 0.2795 |
| R009880 | Lazio | plateforme_en_ligne | total | non | 12 | 0.3004 |

**Raisons possibles de ces erreurs :**

- **Faux positifs** : le cas R002768 (délai de réservation de 526 jours — près d'un an et demi avant l'arrivée) illustre bien le mécanisme : un délai extrême pousse la probabilité à 0.7979 alors que le client maintient en réalité, probablement parce qu'un long délai n'est pas systématiquement synonyme d'incertitude (il peut s'agir d'une réservation planifiée de longue date, via une agence, donc plus engageante qu'une réservation impulsive). Les autres faux positifs partagent un tarif remboursable, une des variables les plus importantes du modèle (Q3/section 5.1) — le modèle semble sur-pondérer ce facteur en l'absence d'autres signaux compensateurs.
- **Faux négatifs** : à l'inverse, R008488 (tarif non remboursable, acompte total, proba de seulement 0.0908) montre les limites d'un modèle qui s'appuie fortement sur `tarif_remboursable`/`type_acompte` : ces conditions, censées être engageantes, n'empêchent pas une annulation réelle. Plusieurs faux négatifs partagent un acompte total et un tarif non remboursable — exactement le profil que le modèle associe le plus fortement à un maintien (Q5), ce qui les rend structurellement difficiles à détecter avec les variables actuelles.

**Piste d'amélioration** : les erreurs des deux catégories convergent vers la même limite — le modèle s'appuie beaucoup sur `tarif_remboursable`/`type_acompte`/`delai_reservation_jours`, mais ces variables ne suffisent pas à elles seules à capter l'intention réelle du client (ex. R002768 avec son délai extrême, ou R008488 malgré un acompte total). Un historique client plus riche (nombre de modifications de la réservation avant l'annulation, historique de correspondance) ou une variable croisant `type_acompte` × `delai_reservation_jours` pourrait aider à distinguer ces cas atypiques du profil moyen.

---

### **6. Conclusion et Recommandations**

Le modèle final (Gradient Boosting, rééquilibré via `sample_weight`, seuil 0.45) atteint un F1 de 0.5000 sur la validation temporelle — un score modeste en valeur absolue mais cohérent avec la difficulté du problème (classe minoritaire à 28 %, comportement client intrinsèquement bruité). Ses principales limites : un taux de faux positifs élevé (653 sur la validation), une performance non vérifiée par région ou type de destination (Q8), et un feature engineering qui n'a pas permis d'aller au-delà de ce score. Le modèle reste néanmoins directement exploitable comme **signal d'aide à la décision graduel** plutôt que comme automate de décision.

**Recommandation opérationnelle finale :**

Déployer `probabilite_annulation` comme un score de priorisation pour l'équipe commerciale (relance ciblée au-delà du seuil 0.45), sans jamais déclencher d'action automatique irréversible (annulation, sur-booking) sur la seule base de la prédiction — cohérent avec la mission du sujet de ne pas pénaliser les clients qui, in fine, maintiennent leur réservation.

---

### **7. Reproductibilité**

- version de Python : ⚠️ **à compléter** (`!python --version` sur Colab)
- principales bibliothèques et versions : ⚠️ **à compléter** (`pandas`, `numpy`, `scikit-learn` — exécute `!pip freeze | grep -E "pandas|numpy|scikit-learn"` sur Colab et colle le résultat)
- graine(s) aléatoire(s) : `random_state=42` fixé sur tous les modèles (`LogisticRegression`, `RandomForestClassifier`, `GradientBoostingClassifier`, `GridSearchCV`)
- commande ou procédure d'exécution : ouvrir `notebook.ipynb` sur Google Colab, exécuter les cellules dans l'ordre (Runtime > Run all), uploader `reservations_train.csv` puis `reservations_test.csv` quand demandé
- durée approximative d'entraînement : ⚠️ **à compléter**
- environnement utilisé : Google Colab

---

### **8. Bibliographie**

- Documentation officielle scikit-learn : `ColumnTransformer`, `Pipeline`, `TimeSeriesSplit`, `GridSearchCV`, `compute_sample_weight` — [scikit-learn.org](https://scikit-learn.org)
- Sujet et données du hackathon : *Atlantic Haven Hotels*, ISPM Madagascar — [github.com/AndryRAB/atlantic-haven](https://github.com/AndryRAB/atlantic-haven)
- **Outil d'IA générative utilisé** : Claude (Anthropic) a été utilisé en support tout au long du projet — aide au débogage du pipeline (dont un bug de `ColumnTransformer` partagé entre plusieurs modèles), génération du code de feature engineering et d'analyse des erreurs (sections 4 à 6 du notebook), diagnostic de l'écart entre le taux d'annulation prédit et observé, et rédaction de ce rapport à partir des résultats produits par l'équipe. Toutes les métriques et sorties de code proviennent de l'exécution réelle du notebook par l'équipe.
- ⚠️ **À compléter** : ajoute ici tout cours, article ou ressource supplémentaire utilisé par l'équipe.
