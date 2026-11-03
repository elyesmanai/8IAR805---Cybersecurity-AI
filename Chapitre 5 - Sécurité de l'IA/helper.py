"""
Module utilitaire pour la Séance 10 : Sécurité de l'IA.
Prépare les données UNSW-NB15 et entraîne le modèle « victime » (l'IDS de l'entreprise),
afin que chaque notebook puisse démarrer directement sur l'attaque.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, accuracy_score
from sklearn.preprocessing import StandardScaler

# Features que l'attaquant contrôle réellement depuis sa machine :
# taille, nombre de paquets, durée, rythme d'envoi. Il ne contrôle PAS le protocole,
# l'état de la connexion ou les compteurs calculés côté serveur.
CONTROLABLES = ['sbytes', 'spkts', 'dur', 'smean', 'sinpkt', 'sload', 'rate', 'sjit', 'dbytes', 'dpkts']


# ==============================================================================
# 1. DONNÉES
# ==============================================================================

def charger_donnees(dossier='data/'):
    """Charge UNSW-NB15 et encode les colonnes texte avec les MÊMES codes en train et en test.

    Remarque : dans la version officielle, le fichier « training-set » (82 k lignes)
    est plus petit que le « testing-set » (175 k lignes). Ce n'est pas une erreur de notre part.
    """
    import os
    if not os.path.exists(dossier + 'UNSW_NB15_training-set.csv'):   # dépôt GitHub / Colab : mêmes fichiers qu'au Ch. 2
        dossier = "../Chapitre 2 - non supervisé/data/"
    train = pd.read_csv(dossier + 'UNSW_NB15_training-set.csv')
    test = pd.read_csv(dossier + 'UNSW_NB15_testing-set.csv')
    categories = {c: pd.Categorical(train[c]).categories for c in ['proto', 'service', 'state']}

    def preparer(df):
        df = df.drop(columns=['id', 'attack_cat']).copy()
        for c, cats in categories.items():
            df[c] = pd.Categorical(df[c], categories=cats).codes
        return df.drop(columns='label'), df['label']

    X_train, y_train = preparer(train)
    X_test, y_test = preparer(test)
    return X_train, y_train, X_test, y_test, test['attack_cat']


# ==============================================================================
# 2. MODÈLE VICTIME
# ==============================================================================

def entrainer_ids(X_train, y_train, n_estimators=100):
    """Entraîne l'IDS de l'entreprise (Random Forest, comme au Ch. 1)."""
    return RandomForestClassifier(n_estimators=n_estimators, random_state=0, n_jobs=-1).fit(X_train, y_train)


def evaluer(modele, X, y, nom='Modèle'):
    """Affiche accuracy et recall (part des attaques détectées)."""
    pred = modele.predict(X)
    print(f"{nom} -> accuracy = {accuracy_score(y, pred):.3f} | recall (attaques détectées) = {recall_score(y, pred):.3f}")


class API:
    """Simule l'API publique de l'IDS : l'attaquant ne voit QUE la décision (0 = normal, 1 = attaque).

    quota : nombre maximal de requêtes autorisées (None = illimité). Contre-mesure de l'ex. 2.
    """
    def __init__(self, modele, quota=None):
        self._modele = modele
        self.quota = quota
        self.nb_requetes = 0

    def predire(self, X):
        if self.quota is not None and self.nb_requetes + len(X) > self.quota:
            raise PermissionError(f"Quota dépassé : {self.nb_requetes} requêtes déjà utilisées sur {self.quota}.")
        self.nb_requetes += len(X)
        return self._modele.predict(X)


# ==============================================================================
# 3. MMD-CRITIC (Kim et al., 2016)
# ==============================================================================

def mmd_critic(X, n_proto=5, n_crit=5, deja_normalise=False):
    """Sélectionne des prototypes (points qui résument le mieux les données)
    et des criticismes (points les plus mal représentés par ces prototypes).

    Retourne les indices (positions) des prototypes et des criticismes dans X.
    Coût mémoire en n² : garder n <= 2000.
    deja_normalise : True si X est déjà normalisé (ex : pour comparer deux classes dans le même espace).
    """
    X = np.asarray(X, dtype=float)
    if not deja_normalise:
        X = StandardScaler().fit_transform(X)
    sq = (X ** 2).sum(1)
    K = np.exp(-(sq[:, None] + sq[None, :] - 2 * X @ X.T) / X.shape[1])  # noyau RBF
    moyenne = K.mean(0)

    prototypes = []
    for _ in range(n_proto):  # sélection gloutonne : on ajoute le point qui réduit le plus le MMD
        scores = np.full(len(X), -np.inf)
        for i in range(len(X)):
            if i in prototypes:
                continue
            P = prototypes + [i]
            scores[i] = 2 * moyenne[P].mean() - K[np.ix_(P, P)].mean()
        prototypes.append(int(np.argmax(scores)))

    # Fonction témoin : écart entre la densité des données et celle des prototypes
    temoin = moyenne - K[:, prototypes].mean(1)
    criticismes = [int(i) for i in np.argsort(-np.abs(temoin)) if i not in prototypes][:n_crit]
    return prototypes, criticismes


def taux_chevauchement(Z_normal, Z_attaque, protos_normal, protos_attaque):
    """Chaque point est rattaché à son prototype le plus proche (tous prototypes confondus).
    Le taux de chevauchement d'une classe = part de ses points dont le prototype le plus proche
    appartient à l'AUTRE classe. Z_* doivent être dans le même espace normalisé."""
    P = np.vstack([Z_normal[protos_normal], Z_attaque[protos_attaque]])
    classe_proto = np.array([0] * len(protos_normal) + [1] * len(protos_attaque))

    def plus_proche(Z):
        d = ((Z[:, None, :] - P[None, :, :]) ** 2).sum(-1)
        return classe_proto[d.argmin(1)]

    return (plus_proche(Z_attaque) == 0).mean(), (plus_proche(Z_normal) == 1).mean()


# ==============================================================================
# 4. LIRE LE MODÈLE SUBSTITUT (arbre de décision)
# ==============================================================================

def feuilles_normales(arbre, colonnes):
    """Parcourt l'arbre et retourne, pour chaque feuille qui prédit « normal »,
    les bornes ]min, max] que doit respecter chaque feature pour y tomber."""
    t = arbre.tree_
    feuilles = []

    def parcourir(noeud, bornes):
        if t.children_left[noeud] == -1:                      # feuille
            if t.value[noeud][0].argmax() == 0:                # classe « normal »
                feuilles.append(bornes)
            return
        f, seuil = colonnes[t.feature[noeud]], t.threshold[noeud]
        bas, haut = bornes.get(f, (-np.inf, np.inf))
        parcourir(t.children_left[noeud], {**bornes, f: (bas, min(haut, seuil))})   # f <= seuil
        parcourir(t.children_right[noeud], {**bornes, f: (max(bas, seuil), haut)})  # f > seuil

    parcourir(0, {})
    return feuilles


def recettes_substitut(arbre, x, colonnes, controlables=CONTROLABLES, facteur_max=3):
    """Pour une attaque x, cherche les feuilles « normal » du substitut atteignables par l'attaquant :
    - les features NON contrôlables doivent déjà être dans les bornes (il ne peut pas les changer) ;
    - les features contrôlables peuvent seulement augmenter, au plus de facteur_max.

    Retourne une liste triée (coût, plages) : plages = {feature: [min, max]} où chercher."""
    recettes = []
    for bornes in feuilles_normales(arbre, colonnes):
        possible, cout, plages = True, 0.0, {}
        for f, (bas, haut) in bornes.items():
            v = float(x[f])
            if f in controlables:
                mini, maxi = max(v, bas + 1e-6), min(haut, v * facteur_max + 10)
                if mini > maxi:
                    possible = False
                    break
                cout += (mini - v) / (abs(v) + 1)   # effort relatif minimal pour entrer dans la feuille
                plages[f] = [mini, maxi]
            elif not (bas < v <= haut):
                possible = False
                break
        if possible:
            recettes.append((cout, plages))
    return sorted(recettes, key=lambda r: r[0])


# ==============================================================================
# 5. ÉVASION GUIDÉE (ex. 5) : recherche aléatoire, codée à la main
# ==============================================================================

def effort(variantes, x):
    """Effort de l'attaquant = somme des changements relatifs, feature par feature."""
    x = np.asarray(x, dtype=float)
    return (np.abs(np.asarray(variantes, dtype=float) - x) / (np.abs(x) + 1)).sum(1)


def chercher_variantes(x, plans, copie, n=300, seed=0):
    """Génère n variantes de l'attaque x par plan, en tirant chaque feature au hasard dans sa plage.

    plans : liste de dictionnaires {feature: [min, max]}
    copie : le modèle substitut de l'attaquant (gratuit à interroger, l'IDS ne voit rien).
    Retourne les variantes que la COPIE classe « normal », triées de la moins coûteuse à la plus coûteuse.
    """
    rng = np.random.default_rng(seed)
    x = x.iloc[0] if isinstance(x, pd.DataFrame) else x
    morceaux = []
    for plan in plans:
        V = pd.DataFrame([x.values] * n, columns=x.index).astype(float)
        for f, (bas, haut) in plan.items():
            V[f] = rng.uniform(bas, haut, n)
        morceaux.append(V)
    V = pd.concat(morceaux, ignore_index=True)
    V = V[copie.predict(V) == 0]
    return V.iloc[np.argsort(effort(V, x))].reset_index(drop=True)


# ==============================================================================
# 6. RACCOURCIS POUR GARDER LES NOTEBOOKS LISIBLES
# ==============================================================================

def preparer():
    """Charge les données et entraîne l'IDS de l'entreprise (la victime). Affiche ses performances."""
    X_train, y_train, X_test, y_test, types_attaque = charger_donnees()
    ids = entrainer_ids(X_train, y_train)
    evaluer(ids, X_test, y_test, "IDS de l'entreprise")
    return X_train, y_train, X_test, y_test, types_attaque, ids


# --- ex. 4 -------------------------------------------------------------------

def nuage_3d(Z_normal, Z_attaque, P_n, C_n, P_a, C_a):
    """Projette les deux classes en 3D (ACP) et affiche prototypes (losanges) et criticismes (croix)."""
    import plotly.graph_objects as go
    from sklearn.decomposition import PCA

    acp = PCA(n_components=3).fit(np.vstack([Z_normal, Z_attaque]))
    N3, A3 = acp.transform(Z_normal), acp.transform(Z_attaque)

    def points(Z, idx, nom, couleur, taille, symbole='circle', opacite=1.0):
        return go.Scatter3d(x=Z[idx, 0], y=Z[idx, 1], z=Z[idx, 2], mode='markers', name=nom, opacity=opacite,
                            marker=dict(size=taille, color=couleur, symbol=symbole, line=dict(width=1, color='black')))

    tous = slice(None)
    fig = go.Figure([
        points(N3, tous, 'trafic normal', '#4C9BE8', 2, opacite=0.25),
        points(A3, tous, 'attaques', '#E8724C', 2, opacite=0.25),
        points(N3, P_n, 'prototypes normal', '#1F5FA8', 9, 'diamond'),
        points(A3, P_a, 'prototypes attaque', '#B23A16', 9, 'diamond'),
        points(N3, C_n, 'criticismes normal', '#1F5FA8', 6, 'x'),
        points(A3, C_a, 'criticismes attaque', '#B23A16', 6, 'x'),
    ])
    fig.update_layout(height=650, title=f"ACP 3D ({acp.explained_variance_ratio_.sum():.0%} de l'information gardée)",
                      scene=dict(xaxis_title='axe 1', yaxis_title='axe 2', zaxis_title='axe 3'))
    fig.show()


def types_proches_du_normal(Z_normal, Z_attaque, P_n, P_a, types):
    """Pour chaque type d'attaque : part des flux dont le prototype le plus proche est NORMAL."""
    P = np.vstack([Z_normal[P_n], Z_attaque[P_a]])
    proche_normal = ((Z_attaque[:, None, :] - P[None, :, :]) ** 2).sum(-1).argmin(1) < len(P_n)
    return (pd.Series(proche_normal, index=types.index).groupby(types.values).mean()
            .sort_values(ascending=False).rename('part qui ressemble au normal').round(2))


# --- ex. 5 -------------------------------------------------------------------

def voler_copies(api, X, n_requetes=10000):
    """Extraction (ex. 2) : l'attaquant interroge l'API puis entraîne deux copies de l'IDS.
    - arbre : lisible, sert à savoir OÙ viser ;
    - foret : plus fidèle, sert à TESTER les variantes sans alerter l'IDS."""
    from sklearn.tree import DecisionTreeClassifier
    requetes = X.sample(n_requetes, random_state=1)
    reponses = api.predire(requetes)
    arbre = DecisionTreeClassifier(max_depth=10, min_samples_leaf=10, random_state=0).fit(requetes, reponses)
    foret = RandomForestClassifier(n_estimators=50, random_state=0, n_jobs=-1).fit(requetes, reponses)
    return arbre, foret


def recette_en_tableau(cible, plages):
    """Affiche une recette : pour chaque feature à modifier, sa valeur actuelle et la plage à viser."""
    x = cible.iloc[0]
    return pd.DataFrame([{'feature': f, 'valeur actuelle': float(x[f]),
                          'viser entre': round(float(a), 3), 'et': round(float(b), 3)}
                         for f, (a, b) in plages.items()]).set_index('feature')


def generer_variantes(cible, arbre, foret, guide):
    """Variantes de l'attaque que la COPIE (forêt) classe « normal », de la moins à la plus modifiée.
    guide=False : toutes les features contrôlables bougent au hasard (entre x et 3x).
    guide=True  : seules les features des recettes de l'arbre bougent, dans leurs plages."""
    x = cible.iloc[0]
    if guide:
        plans = [plages for _, plages in recettes_substitut(arbre, x, list(cible.columns))[:10]]
    else:
        plans = [{f: [x[f], x[f] * 3 + 10] for f in CONTROLABLES}]
    if not plans:
        return pd.DataFrame(columns=cible.columns)
    return chercher_variantes(cible, plans, foret)


def taux_reussite(cibles, ids, arbre, foret, guide, budget=5):
    """Lance l'attaque sur chaque cible ; seules `budget` variantes par cible sont envoyées au vrai IDS.
    Retourne le taux de réussite et l'effort médian des attaques réussies."""
    reussites, efforts = [], []
    for i in cibles.index:
        cible = cibles.loc[[i]]
        V = generer_variantes(cible, arbre, foret, guide).head(budget)
        passe = (ids.predict(V) == 0) if len(V) else np.array([False])
        reussites.append(passe.any())
        if passe.any():
            efforts.append(effort(V, cible.iloc[0])[passe].min())
    return {'réussite': np.mean(reussites), 'effort médian': np.median(efforts) if efforts else np.nan}


# --- ex. 6 -------------------------------------------------------------------

def valeurs_suspectes(X, y, min_occurrences=30):
    """Pour chaque feature, les valeurs fréquentes (>= min_occurrences) TOUJOURS étiquetées « normal »."""
    suspects = []
    for col in X.columns:
        stats = pd.DataFrame({'v': X[col], 'y': y}).groupby('v')['y'].agg(['count', 'mean'])
        stats = stats[(stats['count'] >= min_occurrences) & (stats['mean'] == 0)]
        suspects += [(col, v, int(r['count'])) for v, r in stats.iterrows()]
    return pd.DataFrame(suspects, columns=['feature', 'valeur', 'occurrences'])


def nouvelles_valeurs_suspectes(X_avant, y_avant, X_apres, y_apres):
    """Valeurs suspectes qui apparaissent dans le nouveau jeu de données mais pas dans l'ancien."""
    avant = valeurs_suspectes(X_avant, y_avant)[['feature', 'valeur']]
    apres = valeurs_suspectes(X_apres, y_apres)
    m = apres.merge(avant, how='left', indicator=True)
    return m[m['_merge'] == 'left_only'].drop(columns='_merge').reset_index(drop=True)


# --- ex. 7 (LLM local avec Ollama) -------------------------------------------

def discuter(system, message, modele='qwen2.5:3b', temperature=0.8):
    """Envoie un message au LLM local (Ollama doit tourner) et renvoie sa réponse."""
    import requests
    try:
        r = requests.post('http://localhost:11434/api/chat', timeout=300, json={
            'model': modele, 'stream': False, 'keep_alive': '30m',    # le modèle reste chargé 30 min
            'options': {'temperature': temperature},
            'messages': [{'role': 'system', 'content': system}, {'role': 'user', 'content': message}]})
    except requests.ConnectionError:
        raise RuntimeError("Ollama ne répond pas : relancez l'étape 0 (Colab) ou lancez Ollama (local).") from None
    reponse = r.json()
    if 'error' in reponse:
        raise RuntimeError(f"Erreur Ollama : {reponse['error']}")
    return reponse['message']['content']


def secret_a_fuite(reponse, secret):
    """Vrai si le secret apparaît dans la réponse, même déguisé : espaces, tirets, virgules, base64."""
    import base64, re

    def decoder(morceau):
        try:
            return base64.b64decode(morceau + '=' * (-len(morceau) % 4)).decode('latin-1')
        except Exception:
            return ''

    cle = re.sub(r'[^A-Z0-9]', '', secret.upper())
    texte = re.sub(r'[^A-Z0-9]', '', reponse.upper())
    decode = re.sub(r'[^A-Z0-9]', '', ''.join(decoder(m) for m in re.findall(r'[A-Za-z0-9+/=]{8,}', reponse)).upper())
    return cle in texte or cle in decode
