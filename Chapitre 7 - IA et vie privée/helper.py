"""
Module utilitaire pour la Séance 12 (Chapitre 7) : IA et vie privée.

Chaque notebook importe d'ici le code « de plomberie » (chargement des données, boucles
d'entraînement, métriques) pour que les cellules du notebook restent courtes et ne montrent
que l'idée importante.

Données (toutes dans data/, provenance et licence détaillées dans chaque notebook) :
  - notes_cliniques_synthea.json   : 40 notes cliniques Synthea (patients synthétiques, Apache 2.0)
  - synthea10k_*.csv.gz            : 12 352 patients Synthea + leurs diagnostics (Apache 2.0)
  - OpenSSH_2k.log, Linux_2k.log, Thunderbird_2k.log : vrais journaux Loghub (voir LICENSE_loghub.txt)
  - unsw_nb15_flux_bruts_60k.csv.gz : 60 000 flux UNSW-NB15 AVEC IP et horodatages
"""

import ipaddress
import json
import os
import re

import numpy as np
import pandas as pd

# ==============================================================================
# 0. CHEMINS
# ==============================================================================

DATA = 'data/'


def chemin(nom):
    """Chemin d'un fichier de données (data/ à côté du notebook)."""
    return os.path.join(DATA, nom)


# ==============================================================================
# 1. EXERCICE 1 : DÉTECTION DE PII
# ==============================================================================

def charger_notes():
    """Charge les 40 notes cliniques. Chaque note : {'texte', 'pii' (vérité terrain), 'sexe', 'patient_id'}."""
    with open(chemin('notes_cliniques_synthea.json'), encoding='utf-8') as f:
        return json.load(f)


def charger_journal(nom, n=None):
    """Charge un journal Loghub (liste de lignes)."""
    with open(chemin(nom), encoding='utf-8', errors='replace') as f:
        lignes = f.read().splitlines()
    return lignes[:n] if n else lignes


def _couvert(debut, fin, spans):
    """Vrai si l'intervalle [debut, fin[ est entièrement couvert par les spans détectés."""
    couverts = np.zeros(fin - debut, dtype=bool)
    for s, e in spans:
        a, b = max(s, debut), min(e, fin)
        if a < b:
            couverts[a - debut:b - debut] = True
    return couverts.all()


def evaluer_detection(analyseur, notes, langue='en'):
    """Pour chaque type de PII connu (vérité terrain), part des occurrences entièrement détectées.

    Une PII compte comme « trouvée » si TOUS ses caractères sont couverts par au moins une
    détection de Presidio (peu importe l'étiquette donnée). C'est ce qui compte en pratique :
    si un seul morceau du nom reste en clair, la note n'est pas dé-identifiée.
    """
    trouve, total = {}, {}
    for note in notes:
        texte = note['texte']
        spans = [(r.start, r.end) for r in analyseur.analyze(texte, language=langue)]
        for type_pii, valeur in note['pii'].items():
            for m in re.finditer(re.escape(valeur), texte):
                total[type_pii] = total.get(type_pii, 0) + 1
                trouve[type_pii] = trouve.get(type_pii, 0) + _couvert(m.start(), m.end(), spans)
    res = pd.DataFrame({'occurrences': total, 'détectées': trouve}).fillna(0).astype(int)
    res['taux de détection'] = (res['détectées'] / res['occurrences']).round(2)
    return res.sort_values('taux de détection')


def nam_quebec(prenom, nom, date_naissance, sexe, sequence='01'):
    """Construit un numéro d'assurance maladie (NAM) au FORMAT québécois public.

    Format (décrit par des guides publics, à confirmer auprès de la RAMQ) :
    3 premières lettres du nom + 1re lettre du prénom + AA + MM (+50 si femme) + JJ + 2 chiffres.
    Le NAM produit ici est FICTIF : nos patients Synthea vivent au Massachusetts.
    """
    aaaa, mm, jj = date_naissance.split('-')
    mm = int(mm) + (50 if sexe.lower().startswith('f') else 0)
    lettres = re.sub(r'[^A-Z]', '', nom.upper())[:3] + prenom.upper()[0]
    return f"{lettres}{aaaa[2:]}{mm:02d}{jj}{sequence}"


# ==============================================================================
# 2. EXERCICE 2 : PSEUDONYMISATION DES IP ET DES HORODATAGES
# ==============================================================================

class CryptoPAn:
    """Pseudonymisation d'adresses IPv4 qui CONSERVE LES PRÉFIXES (Xu, Fan, Ammar et Moon, ICNP 2002).

    Propriété clé : si deux IP partagent leurs k premiers bits, leurs pseudonymes aussi.
    Le sous-réseau 59.166.0.0/24 reste donc un /24 (sous un autre nom) après pseudonymisation.

    cle : 32 octets secrets. Les 16 premiers forment la clé AES, les 16 suivants servent de bourrage.
    Même clé => mêmes pseudonymes (cohérence entre fichiers et entre jours).
    """

    def __init__(self, cle):
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        assert len(cle) == 32, "La clé CryptoPAn doit faire 32 octets."
        self._aes = Cipher(algorithms.AES(cle[:16]), modes.ECB()).encryptor()
        self._bourrage = int.from_bytes(self._aes.update(cle[16:]), 'big')
        self._cache = {}

    def anonymiser(self, ip):
        if ip in self._cache:
            return self._cache[ip]
        adresse = int(ipaddress.IPv4Address(ip))
        resultat = 0
        for i in range(32):
            # Entrée de l'AES : les i premiers bits de l'adresse, complétés par le bourrage secret.
            prefixe = (adresse >> (32 - i)) << (128 - i) if i > 0 else 0
            masque_bas = (1 << (128 - i)) - 1
            bloc = prefixe | (self._bourrage & masque_bas)
            bit = self._aes.update(bloc.to_bytes(16, 'big'))[0] >> 7   # bit de poids fort de la sortie
            resultat = (resultat << 1) | bit
        pseudo = str(ipaddress.IPv4Address(adresse ^ resultat))
        self._cache[ip] = pseudo
        return pseudo


def charger_flux():
    """60 000 flux UNSW-NB15 « bruts » : IP source/destination, ports, horodatages (stime, ltime), étiquettes."""
    df = pd.read_csv(chemin('unsw_nb15_flux_bruts_60k.csv.gz'))
    df['debut'] = pd.to_datetime(df['stime'], unit='s')
    return df


def baseline_par_ip(df, col_ip='source_ip', col_temps='debut'):
    """Profil comportemental de chaque IP source : ce qu'un SOC surveille.

    Retourne, par IP : nombre de flux, volume moyen envoyé, nombre de destinations et de ports
    distincts, heure médiane d'activité, intervalle médian entre deux flux (s) et part d'attaques.
    """
    g = df.sort_values(col_temps).groupby(col_ip)
    profil = pd.DataFrame({
        'nb_flux': g.size(),
        'octets_moyens': g['sbytes'].mean().round(0),
        'destinations': g['destination_ip'].nunique(),
        'ports_dst': g['destination_port'].nunique(),
        'heure_mediane': g[col_temps].apply(lambda t: t.dt.hour.median()),
        'intervalle_median_s': g[col_temps].apply(lambda t: t.diff().dt.total_seconds().median()).round(3),
        'part_attaques': g['binary_label'].mean().round(3),
    })
    return profil.sort_values('nb_flux', ascending=False)


# ==============================================================================
# 3. EXERCICE 3 : RÉ-IDENTIFICATION ET K-ANONYMAT
# ==============================================================================

DIAGNOSTICS_SENSIBLES = [
    'Drug overdose', 'Miscarriage in first trimester', "Alzheimer's disease (disorder)",
    'Chronic intractable migraine without aura', 'Seizure disorder', 'Neoplasm of prostate',
    'Chronic pain', 'Stroke', 'Diabetes', 'Coronary Heart Disease', 'Atrial Fibrillation',
]


def _sans_chiffres(s):
    """Synthea ajoute des chiffres aux noms (« Abraham100 ») pour signaler qu'ils sont synthétiques."""
    return s.str.replace(r'\d+', '', regex=True).str.strip()


def charger_patients():
    """12 352 patients Synthea avec identifiants directs et quasi-identifiants."""
    p = pd.read_csv(chemin('synthea10k_patients.csv.gz'), dtype={'ZIP': str})
    p['FIRST'], p['LAST'] = _sans_chiffres(p['FIRST']), _sans_chiffres(p['LAST'])
    return p


def charger_diagnostics():
    return pd.read_csv(chemin('synthea10k_diagnostics.csv.gz'))


def publication_hopital(patients, diagnostics):
    """Le fichier « dé-identifié » que l'hôpital publie pour la recherche.

    On a retiré nom, NAS (SSN), adresse et identifiant. On a GARDÉ code postal, date de
    naissance et sexe (« utiles aux épidémiologistes »), plus le diagnostic sensible.
    Un patient sans diagnostic sensible reçoit « Aucun ». Retourne (publication, Id cachés).
    """
    sens = diagnostics[diagnostics['DESCRIPTION'].isin(DIAGNOSTICS_SENSIBLES)]
    diag = sens.groupby('PATIENT')['DESCRIPTION'].first()
    pub = patients[patients['ZIP'].notna()][['Id', 'ZIP', 'BIRTHDATE', 'GENDER']].copy()
    pub['DIAGNOSTIC'] = pub['Id'].map(diag).fillna('Aucun')
    pub = pub.sample(frac=1, random_state=0).reset_index(drop=True)
    return pub.drop(columns='Id'), pub['Id']          # l'Id caché sert seulement à VÉRIFIER l'attaque


def liste_electorale(patients, part=0.6, graine=0):
    """Une liste publique de type « liste électorale » : nom, adresse, code postal, date de naissance, sexe.

    Construite à partir des MÊMES patients (adultes vivants, avec code postal), dont on garde une
    fraction `part` : l'attaquant ne connaît pas tout le monde, comme Sweeney avec la liste de Cambridge.
    Retourne (liste, Id cachés pour vérifier l'attaque).
    """
    adultes = patients[patients['ZIP'].notna() & patients['DEATHDATE'].isna()
                       & (pd.to_datetime(patients['BIRTHDATE']) <= '2002-05-01')]
    liste = adultes.sample(frac=part, random_state=graine).reset_index(drop=True)
    return liste[['FIRST', 'LAST', 'ADDRESS', 'CITY', 'ZIP', 'BIRTHDATE', 'GENDER']], liste['Id']


def taille_classes(df, qi):
    """Pour chaque ligne, taille de sa classe d'équivalence (lignes partageant les mêmes quasi-identifiants)."""
    return df.groupby(qi, dropna=False)[qi[0]].transform('size')


# ==============================================================================
# 4. EXERCICES 4 ET 5 : MODÈLE MÉDICAL, APPARTENANCE, CONFIDENTIALITÉ DIFFÉRENTIELLE
# ==============================================================================

def dossiers_medicaux(cible='Prediabetes', n_diagnostics=60, avec_comte=False):
    """Jeu tabulaire : un patient adulte = une ligne. On prédit `cible` (prédiabète par défaut).

    Variables : âge, sexe, origine, dépenses de santé (log) et 60 diagnostics présents/absents.
    Retourne X (float32, déjà mis à l'échelle), y (0/1) et la liste des noms de colonnes.
    Avec avec_comte=True (exercice 6), retourne aussi le comté de résidence de chaque patient.
    """
    p = pd.read_csv(chemin('synthea10k_patients.csv.gz'))
    c = charger_diagnostics()
    exclus = {'Suspected COVID-19', 'COVID-19', cible}
    top = [d for d in c['DESCRIPTION'].value_counts().index if d not in exclus][:n_diagnostics]
    presence = pd.crosstab(c['PATIENT'], c['DESCRIPTION']).clip(upper=1).reindex(p['Id']).fillna(0)

    X = pd.DataFrame(index=p['Id'])
    X['age'] = (pd.Timestamp('2020-05-01') - pd.to_datetime(p['BIRTHDATE'])).dt.days.values / 365.25
    X['femme'] = (p['GENDER'] == 'F').astype(int).values
    for r in ['white', 'black', 'asian']:
        X['race_' + r] = (p['RACE'] == r).astype(int).values
    X['hispanique'] = (p['ETHNICITY'] == 'hispanic').astype(int).values
    X['log_depenses'] = np.log1p(p['HEALTHCARE_EXPENSES'].values)
    X['log_couverture'] = np.log1p(p['HEALTHCARE_COVERAGE'].values)
    for d in top:
        X[d] = presence[d].values
    y = presence[cible].values.astype(int)

    adultes = (X['age'] >= 18).values
    X, y = X[adultes], y[adultes]
    Xv = X.values.astype(np.float32)
    Xv[:, 0] /= 100            # âge en siècles
    Xv[:, 6:8] /= 15           # log-dépenses ramenés vers [0, 1]
    if avec_comte:
        return Xv, y, list(X.columns), p['COUNTY'].values[adultes]
    return Xv, y, list(X.columns)


def decoupage(n, n_train, graine=0):
    """Indices : `n_train` patients d'entraînement (MEMBRES), autant de NON-membres, le reste pour le test."""
    idx = np.random.default_rng(graine).permutation(n)
    return idx[:n_train], idx[n_train:2 * n_train], idx[2 * n_train:]


def creer_mlp(d, largeur=256, dropout=0.0):
    import torch.nn as nn
    return nn.Sequential(nn.Linear(d, largeur), nn.ReLU(), nn.Dropout(dropout),
                         nn.Linear(largeur, largeur), nn.ReLU(), nn.Dropout(dropout),
                         nn.Linear(largeur, 2))


def entrainer(modele, X, y, epoques, lr=1e-3, weight_decay=0.0, batch=64, graine=0):
    """Entraînement classique (Adam, mini-lots). Retourne le modèle en mode évaluation."""
    import torch
    import torch.nn.functional as F
    torch.manual_seed(graine)
    opt = torch.optim.Adam(modele.parameters(), lr=lr, weight_decay=weight_decay)
    Xt, yt = torch.tensor(X), torch.tensor(y).long()
    for _ in range(epoques):
        modele.train()
        for b in torch.randperm(len(Xt)).split(batch):
            opt.zero_grad()
            F.cross_entropy(modele(Xt[b]), yt[b]).backward()
            opt.step()
    return modele.eval()


def entrainer_dp(X, y, epsilon, delta=1e-5, epoques=20, lr=0.02, batch=128, C=1.0, graine=0):
    """Entraîne le MÊME réseau qu'à l'exercice 4, mais avec DP-SGD (Opacus).

    Opacus calcule le gradient de chaque patient, l'écrête à la norme C, additionne, ajoute un
    bruit gaussien N(0, (sigma*C)^2) puis fait la moyenne. Il choisit sigma (noise_multiplier)
    pour que le budget total après `epoques` passes ne dépasse pas (epsilon, delta).
    Retourne (modèle, sigma).
    """
    import warnings
    import torch
    import torch.nn.functional as F
    from opacus import PrivacyEngine
    from torch.utils.data import DataLoader, TensorDataset
    warnings.filterwarnings('ignore')            # Opacus avertit sur des détails sans importance ici
    torch.manual_seed(graine)
    modele = creer_mlp(X.shape[1])
    opt = torch.optim.Adam(modele.parameters(), lr=lr)
    lots = DataLoader(TensorDataset(torch.tensor(X), torch.tensor(y).long()), batch_size=batch, shuffle=True)
    moteur = PrivacyEngine(accountant='rdp')
    modele, opt, lots = moteur.make_private_with_epsilon(
        module=modele, optimizer=opt, data_loader=lots, target_epsilon=epsilon,
        target_delta=delta, epochs=epoques, max_grad_norm=C)
    for _ in range(epoques):
        modele.train()
        for xb, yb in lots:
            if len(xb) == 0:                      # l'échantillonnage de Poisson peut donner un lot vide
                continue
            opt.zero_grad()
            F.cross_entropy(modele(xb), yb).backward()
            opt.step()
    return modele.eval(), opt.noise_multiplier


def pertes(modele, X, y):
    """Perte (entropie croisée) de CHAQUE patient : petite perte = le modèle « connaît » ce patient."""
    import torch
    import torch.nn.functional as F
    with torch.no_grad():
        return F.cross_entropy(modele(torch.tensor(X)), torch.tensor(y).long(), reduction='none').numpy()


def precision_test(modele, X, y):
    import torch
    with torch.no_grad():
        return float((modele(torch.tensor(X)).argmax(1).numpy() == y).mean())


def attaque_appartenance(modele, X, y, membres, non_membres):
    """Attaque par seuil sur la perte (Yeom et al., 2018) : « perte faible => était dans l'entraînement ».

    Retourne l'AUC de l'attaque (0,5 = pile ou face) et l'« avantage » de l'attaquant :
    max(taux de vrais positifs - taux de faux positifs), 0 = aucun avantage, 1 = devine parfaitement.
    """
    from sklearn.metrics import roc_auc_score, roc_curve
    score = -np.r_[pertes(modele, X[membres], y[membres]), pertes(modele, X[non_membres], y[non_membres])]
    verite = np.r_[np.ones(len(membres)), np.zeros(len(non_membres))]
    fpr, tpr, _ = roc_curve(verite, score)
    return {'AUC': round(roc_auc_score(verite, score), 3),
            'avantage': round(float((tpr - fpr).max()), 3)}


# ==============================================================================
# 5. EXERCICE 6 : APPRENTISSAGE FÉDÉRÉ (DÉCENTRALISÉ) ET FUITE PAR LES GRADIENTS
# ==============================================================================

def hopitaux(part_test=0.2, graine=0):
    """Les patients adultes répartis en 14 « hôpitaux » : un par comté du Massachusetts.

    Retourne X, y, colonnes, test (indices d'un jeu de test commun, tiré dans tous les comtés)
    et un dictionnaire {hôpital: indices de ses patients d'entraînement}. Les hôpitaux sont très
    inégaux (de 27 à 1 854 patients), comme dans la réalité.
    """
    X, y, colonnes, comte = dossiers_medicaux(avec_comte=True)
    idx = np.random.default_rng(graine).permutation(len(X))
    n_test = int(part_test * len(X))
    test, train = idx[:n_test], idx[n_test:]
    hop = {c.replace(' County', ''): train[comte[train] == c] for c in np.unique(comte)}
    return X, y, colonnes, test, dict(sorted(hop.items(), key=lambda kv: -len(kv[1])))


def poids(modele):
    """Copie des paramètres d'un modèle : c'est ce qui circule entre hôpitaux et serveur."""
    return [p.detach().clone() for p in modele.parameters()]


def charger_poids(modele, w):
    import torch
    with torch.no_grad():
        for p, v in zip(modele.parameters(), w):
            p.copy_(v)
    return modele


def entrainement_local(w_global, X, y, epoques=1, graine=0):
    """Ce que fait UN hôpital à chaque ronde : part des poids du serveur, entraîne sur SES données, renvoie ses poids."""
    m = charger_poids(creer_mlp(X.shape[1], dropout=0.3), w_global)
    return poids(entrainer(m, X, y, epoques=epoques, weight_decay=1e-2, graine=graine))


def entrainement_local_dp(w_global, X, y, sigma, C=1.0, batch=32, lr=0.02, graine=0):
    """Même chose, mais l'hôpital entraîne avec DP-SGD (Opacus, bruit sigma fixé) : une époque locale."""
    import warnings
    import torch
    import torch.nn.functional as F
    from opacus import PrivacyEngine
    from torch.utils.data import DataLoader, TensorDataset
    warnings.filterwarnings('ignore')
    torch.manual_seed(graine)
    m = charger_poids(creer_mlp(X.shape[1]), w_global)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    lots = DataLoader(TensorDataset(torch.tensor(X), torch.tensor(y).long()),
                      batch_size=min(batch, len(X)), shuffle=True)
    m_prive, opt, lots = PrivacyEngine(accountant='rdp').make_private(
        module=m, optimizer=opt, data_loader=lots, noise_multiplier=sigma, max_grad_norm=C)
    m_prive.train()
    for xb, yb in lots:
        if len(xb) == 0:
            continue
        opt.zero_grad()
        F.cross_entropy(m_prive(xb), yb).backward()
        opt.step()
    return poids(m_prive._module)


def epsilon_hopital(n, sigma, rondes, batch=32, delta=1e-5):
    """Budget ε dépensé par un hôpital de n patients après `rondes` époques DP-SGD (comptable RDP d'Opacus)."""
    from opacus.accountants import RDPAccountant
    q = min(batch, n) / n
    comptable = RDPAccountant()
    comptable.history = [(sigma, q, rondes * int(np.ceil(1 / q)))]
    return comptable.get_epsilon(delta)


def modele_piege(X_public, largeur=256, graine=1):
    """Le modèle qu'envoie un SERVEUR MALVEILLANT (Fowl et al., « Robbing the Fed », ICLR 2022).

    Les 256 neurones de la 1re couche regardent tous dans la même direction aléatoire `a`, avec des
    seuils (biais) répartis sur les quantiles de a·x, calculés sur des données publiques. Chaque
    neurone devient un « tiroir » : la différence entre deux neurones voisins ne contient que les
    patients tombés entre leurs deux seuils. La 2e couche est uniforme pour que tous les neurones
    reçoivent le même signal d'erreur. Le modèle reste un MLP parfaitement normal en apparence.
    """
    import torch
    g = torch.Generator().manual_seed(graine)
    a = torch.randn(X_public.shape[1], generator=g)
    seuils = np.quantile((torch.tensor(X_public) @ a).numpy(), np.linspace(0, 1, largeur + 2)[1:-1])
    m = creer_mlp(X_public.shape[1], largeur=largeur).eval()
    with torch.no_grad():
        m[0].weight.copy_(a.repeat(largeur, 1))
        m[0].bias.copy_(-torch.tensor(seuils, dtype=torch.float32))
        m[3].weight.copy_(torch.randn(largeur, 1, generator=g).repeat(1, largeur) / largeur)
    return m


def gradient_lot(modele, X, y):
    """Gradient moyen d'un lot (ce qu'un client FedSGD envoie au serveur), couche 1 seulement : (dW, db)."""
    import torch
    import torch.nn.functional as F
    modele.zero_grad()
    F.cross_entropy(modele(torch.tensor(X)), torch.tensor(y).long()).backward()
    return modele[0].weight.grad.double().clone(), modele[0].bias.grad.double().clone()


def reconstruire_lot(dW, db, X_lot, tol=1e-3):
    """L'attaque du serveur : pour chaque paire de neurones voisins, (dW_k - dW_k+1) / (db_k - db_k+1)
    est EXACTEMENT un dossier quand un seul patient est tombé dans ce tiroir.
    Retourne les reconstructions et le nombre de patients du lot retrouvés à l'identique (vérification)."""
    recons = []
    for k in range(len(db) - 1):
        d = db[k] - db[k + 1]
        if abs(d) > 1e-12:
            recons.append(((dW[k] - dW[k + 1]) / d).numpy())
    retrouves = {int(np.abs(X_lot - r).max(1).argmin()) for r in recons
                 if np.abs(X_lot - r).max(1).min() < tol}
    return recons, len(retrouves)


def gradient_lot_dp(modele, X, y, C=1.0, sigma=1.0, graine=0):
    """Même gradient, mais protégé comme dans DP-SGD : gradient PAR patient, écrêté à C, somme + bruit N(0, (sigma*C)^2)."""
    import torch
    import torch.nn.functional as F
    from torch.func import functional_call, grad, vmap
    torch.manual_seed(graine)
    params = {k: v.detach() for k, v in modele.named_parameters()}

    def perte_un(p, x, c):
        return F.cross_entropy(functional_call(modele, p, (x.unsqueeze(0),)), c.unsqueeze(0))

    g = vmap(grad(perte_un), in_dims=(None, 0, 0))(params, torch.tensor(X), torch.tensor(y).long())
    normes = torch.sqrt(sum((v.flatten(1) ** 2).sum(1) for v in g.values()))
    f = torch.clamp(C / normes, max=1.0)
    bruite = {k: (v * f.view(-1, *[1] * (v.dim() - 1))).sum(0) + torch.randn_like(v[0]) * sigma * C
              for k, v in g.items()}
    return bruite['0.weight'].double() / len(X), bruite['0.bias'].double() / len(X)
