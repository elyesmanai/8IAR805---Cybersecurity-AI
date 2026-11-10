"""
Module utilitaire pour la Séance 11 : Chaîne d'approvisionnement des modèles.

Deux usages :
1. Importé depuis les notebooks (`from helper import ...`).
2. Exécuté comme script isolé (`python -I helper.py <commande> <fichier>`) pour lire
   les fichiers téléchargés qui ne sont PAS de confiance. Le drapeau -I (mode isolé) empêche
   Python d'importer du code depuis le dossier courant ou celui du script, et ignore
   les variables d'environnement PYTHON*. Ce script ne fait QUE de l'analyse statique :
   il ne désérialise jamais le fichier (pas de pickle.load, pas de torch.load).

Règle du chapitre : on n'exécute qu'un seul pickle piégé, celui qu'on fabrique soi-même
avec une charge utile inoffensive (un print).
"""

import hashlib
import json
import os
import subprocess
import sys

DOSSIER = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(DOSSIER, 'data')
NON_FIABLE = os.path.join(DATA, 'non_fiable')      # tout ce qui vient d'Internet atterrit ici, et nulle part ailleurs
MODELES = os.path.join(DATA, 'modeles')            # nos propres modèles (Ch. 1), donc de confiance


# ==============================================================================
# 1. ANALYSE STATIQUE ISOLÉE (partie exécutée avec python -I)
# ==============================================================================

def _octets_pickle(chemin):
    """Renvoie le flux pickle à analyser, SANS le désérialiser.
    Un fichier PyTorch (.bin, .pt) est une archive ZIP qui contient un fichier data.pkl."""
    import zipfile
    if zipfile.is_zipfile(chemin):
        with zipfile.ZipFile(chemin) as z:
            nom = [n for n in z.namelist() if n.endswith('data.pkl')][0]
            return z.read(nom), f'archive ZIP PyTorch -> {nom}'
    with open(chemin, 'rb') as f:
        return f.read(), 'pickle brut'


def _imports_pickle(chemin):
    """Liste les « module.fonction » que le pickle demande d'importer, en lisant les opcodes un par un.
    pickletools.genops lit le flux sans rien exécuter."""
    import pickletools
    donnees, nature = _octets_pickle(chemin)
    imports, empiles, memo, erreur = [], [], {}, None
    try:
        for op, arg, pos in pickletools.genops(donnees):
            if op.name in ('GLOBAL', 'INST'):
                imports.append(arg.replace(' ', '.'))
            elif op.name in ('SHORT_BINUNICODE', 'BINUNICODE', 'UNICODE', 'BINUNICODE8'):
                empiles.append(arg)                                   # une chaîne posée sur la pile
            elif op.name == 'MEMOIZE':
                memo[len(memo)] = empiles[-1] if empiles else None    # mise en mémoire de la dernière valeur
            elif op.name in ('PUT', 'BINPUT', 'LONG_BINPUT'):
                memo[arg] = empiles[-1] if empiles else None
            elif op.name in ('GET', 'BINGET', 'LONG_BINGET'):
                empiles.append(memo.get(arg))                         # relecture d'une valeur mémorisée
            elif op.name == 'STACK_GLOBAL':
                module, nom = empiles[-2], empiles[-1]                # « importe module.nom »
                imports.append(f'{module}.{nom}')
                empiles.append(None)
            else:
                empiles.append(None)
    except Exception as e:                       # fichier corrompu : on garde ce qu'on a lu AVANT l'erreur
        erreur = f'{type(e).__name__}: {e}'
    return {'nature': nature, 'taille_pickle': len(donnees), 'imports': sorted(set(imports)), 'erreur': erreur}


def _entete(chemin):
    with open(chemin, 'rb') as f:
        debut = f.read(16)
    if debut[:2] == b'PK':
        nature = 'ZIP (format PyTorch)'
    elif debut[:1] == b'\x80':
        nature = f'pickle (protocole {debut[1]})'
    elif len(debut) >= 8 and debut[8:9] == b'{':
        nature = 'safetensors (en-tête JSON)'
    else:
        nature = 'inconnu'
    return {'octets': debut.hex(' '), 'nature': nature, 'taille': os.path.getsize(chemin)}


def _sha256(chemin):
    h = hashlib.sha256()
    with open(chemin, 'rb') as f:
        for bloc in iter(lambda: f.read(1 << 20), b''):
            h.update(bloc)
    return {'sha256': h.hexdigest()}


def _entete_safetensors(chemin):
    """Un fichier safetensors = 8 octets (taille de l'en-tête) + un en-tête JSON + les octets bruts des tenseurs.
    On lit l'en-tête comme du simple texte JSON : aucun code n'est exécuté."""
    import struct
    with open(chemin, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        if n > os.path.getsize(chemin):
            return {'erreur': "ce n'est pas un fichier safetensors"}
        entete = json.loads(f.read(n))
    meta = entete.pop('__metadata__', {})
    tenseurs = {k: {'dtype': v['dtype'], 'shape': v['shape']} for k, v in entete.items()}
    return {'taille_entete': n, 'metadata': meta, 'nb_tenseurs': len(tenseurs), 'tenseurs': tenseurs}


def _ajouter_site_utilisateur():
    """Le mode -I retire les paquets installés avec `pip install --user`. On rajoute ce dossier
    (et lui seul) pour retrouver fickling et modelscan. Le dossier courant reste exclu."""
    import site
    chemin = site.getusersitepackages()
    if os.path.isdir(chemin) and chemin not in sys.path:
        sys.path.append(chemin)


def _fickling(chemin):
    _ajouter_site_utilisateur()
    import fickling
    from fickling.fickle import Pickled
    from fickling.analysis import check_safety
    donnees, _ = _octets_pickle(chemin)
    resultat = {'sur_selon_fickling': None, 'verdict': None, 'alertes': [], 'code_decompile': None}
    try:
        p = Pickled.load(donnees, fail_on_decode_error=False)   # comme le ferait un attaquant : on lit malgré la corruption
        analyse = check_safety(p)
        resultat['verdict'] = str(analyse.severity).split('.')[-1]
        resultat['sur_selon_fickling'] = bool(analyse)
        resultat['alertes'] = sorted({r.message for r in analyse.results if r.message})[:6]
        try:
            import ast
            resultat['code_decompile'] = ast.unparse(p.ast)[:600]
        except Exception as e:
            resultat['code_decompile'] = f'(décompilation impossible : {type(e).__name__})'
    except Exception as e:
        resultat['verdict'] = f'erreur : {type(e).__name__}: {e}'
    return resultat


def _modelscan(chemin):
    """Lance modelscan en mode rapport JSON, dans ce même processus isolé."""
    _ajouter_site_utilisateur()
    import contextlib, io, tempfile
    from modelscan.cli import cli
    with tempfile.TemporaryDirectory() as tmp:
        sortie = os.path.join(tmp, 'rapport.json')
        tampon = io.StringIO()
        try:
            with contextlib.redirect_stdout(tampon), contextlib.redirect_stderr(tampon):
                cli.main(['scan', '-p', chemin, '-r', 'json', '-o', sortie], standalone_mode=False)
        except SystemExit:
            pass
        except Exception as e:
            return {'erreur_modelscan': f'{type(e).__name__}: {e}'}
        if not os.path.exists(sortie):
            return {'erreur_modelscan': tampon.getvalue()[-300:]}
        rapport = json.load(open(sortie))
    return {
        'problemes': [{'gravite': p.get('severity'), 'operateur': f"{p.get('module')}.{p.get('operator')}"}
                      for p in rapport.get('issues', [])],
        'erreurs': [e.get('description', str(e))[:120] for e in rapport.get('errors', [])],
        'resume': {k: v for k, v in rapport.get('summary', {}).items() if k in ('total_issues', 'total_issues_by_severity')},
    }


COMMANDES = {'entete': _entete, 'sha256': _sha256, 'imports': _imports_pickle,
             'fickling': _fickling, 'modelscan': _modelscan, 'safetensors': _entete_safetensors}


def analyser_isole(commande, chemin):
    """Lance `python -I helper.py <commande> <chemin>` dans un processus séparé et renvoie le résultat (dict).
    Le notebook ne touche jamais lui-même au contenu d'un fichier non fiable."""
    proc = subprocess.run([sys.executable, '-I', os.path.join(DOSSIER, 'helper.py'), commande, os.path.abspath(chemin)],
                          capture_output=True, text=True, cwd=DOSSIER, timeout=300)
    if proc.returncode != 0:
        return {'erreur_processus': proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else 'inconnue'}
    return json.loads(proc.stdout)


# ==============================================================================
# 2. TÉLÉCHARGEMENTS DEPUIS HUGGING FACE (sans jamais charger les modèles)
# ==============================================================================

# (dépôt, fichier, description) : de vrais fichiers hébergés sur Hugging Face.
FICHIERS_HF = [
    ('hf-internal-testing/tiny-random-bert', 'pytorch_model.bin', 'Petit BERT (format PyTorch, sain)'),
    ('scikit-learn/tabular-playground', 'model.pkl', 'Pipeline scikit-learn (pickle, sain)'),
    ('mcpotato/42-eicar-street', 'danger.dat', 'Fichier de test de HF : pickle qui exécute du code'),
    ('mcpotato/42-eicar-street', 'model_broken_X.pkl', 'Fichier de test de HF : pickle volontairement « cassé »'),
]

# Repli hors-ligne : copie octet pour octet des deux fichiers de test de mcpotato/42-eicar-street
# (dépôt de test maintenu par un ingénieur de Hugging Face). Charges utiles inoffensives.
_REPLI = {
    'danger.dat': '80049537000000000000008c086275696c74696e73948c046576616c9493948c1b65786563282727277072696e7428276861636b336427292727272994859452942e',
    'model_broken_X.pkl': '8002636275696c74696e730a657865630a28583e0000000a66203d206f70656e28276d795f66696c652e747874272c20276127293b20662e777269746528274d616c6963696f757327293b20662e636c6f736528297452582e',
}


def _silence_hf():
    """Masque les barres de progression et l'avertissement « requêtes non authentifiées » (sans conséquence ici)."""
    import warnings
    os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
    warnings.filterwarnings('ignore', module='tqdm')
    from huggingface_hub.utils import logging as hf_logging
    hf_logging.set_verbosity_error()


def telecharger_hf(fichiers=FICHIERS_HF):
    """Télécharge les fichiers dans data/non_fiable/<dépôt>/ et renvoie la liste des chemins.
    Télécharger n'exécute rien : le danger commence au chargement (pickle.load, torch.load, joblib.load)."""
    _silence_hf()
    from huggingface_hub import hf_hub_download
    os.makedirs(NON_FIABLE, exist_ok=True)
    chemins = []
    for depot, fichier, desc in fichiers:
        dossier = os.path.join(NON_FIABLE, depot.replace('/', '__'))
        try:
            chemin = hf_hub_download(depot, fichier, local_dir=dossier)
        except Exception as e:
            if fichier in _REPLI:
                os.makedirs(dossier, exist_ok=True)
                chemin = os.path.join(dossier, fichier)
                with open(chemin, 'wb') as f:
                    f.write(bytes.fromhex(_REPLI[fichier]))
                print(f'[REPLI] {depot}/{fichier} : réseau indisponible ({type(e).__name__}), copie locale identique utilisée.')
            else:
                print(f'[REPLI] {depot}/{fichier} : réseau indisponible ({type(e).__name__}), fichier ignoré.')
                continue
        chemins.append({'depot': depot, 'fichier': fichier, 'description': desc, 'chemin': chemin})
    return chemins


def verdict_hf(depots):
    """Demande à Hugging Face le résultat de SES propres analyses (picklescan + antivirus) pour chaque dépôt.
    On n'a rien à télécharger pour ça."""
    import pandas as pd
    _silence_hf()
    from huggingface_hub import HfApi
    api, lignes = HfApi(), []
    for depot in depots:
        try:
            info = api.model_info(depot, securityStatus=True)
            statut = getattr(info, 'security_repo_status', None) or {}
            problemes = statut.get('filesWithIssues', [])
            lignes.append({'dépôt': depot,
                           'analyse faite': statut.get('scansDone'),
                           'fichiers signalés': ', '.join(f"{p['path']} ({p['level']})" for p in problemes) or 'aucun'})
        except Exception as e:
            lignes.append({'dépôt': depot, 'analyse faite': None, 'fichiers signalés': f'[REPLI] indisponible ({type(e).__name__})'})
    return pd.DataFrame(lignes)


# ==============================================================================
# 3. DONNÉES ET MODÈLES DU CH. 1 (détection de maliciels à partir des en-têtes PE)
# ==============================================================================

def charger_maliciels():
    """Le jeu de données du Ch. 1 : 19 611 exécutables Windows décrits par 77 champs de l'en-tête PE."""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    chemin = os.path.join(DATA, 'dataset_malware.csv')
    if not os.path.exists(chemin):                      # dépôt GitHub / Colab : même fichier qu'au Ch. 1
        chemin = os.path.join(DOSSIER, '..', 'Chapitre 1 - supervisé tabulaire', 'data', 'dataset_malware.csv')
    df = pd.read_csv(chemin)
    X = df.drop(columns=['Name', 'Malware']).astype('float32')
    y = df['Malware']
    return train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)


def entrainer_modeles(X_train, y_train):
    """Les trois modèles du Ch. 1 : arbre de décision, forêt aléatoire, KNN."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    modeles = {
        'arbre': DecisionTreeClassifier(max_depth=8, random_state=0),
        'foret': RandomForestClassifier(n_estimators=50, max_depth=12, random_state=0, n_jobs=-1),
        'knn': KNeighborsClassifier(n_neighbors=5),
    }
    for m in modeles.values():
        m.fit(X_train, y_train)
    return modeles


# ==============================================================================
# 4. INTÉGRITÉ : EMPREINTES ET SIGNATURES
# ==============================================================================

def sha256(chemin):
    """Empreinte SHA-256 d'un fichier de confiance (lu par blocs de 1 Mo)."""
    return _sha256(chemin)['sha256']


def ecrire_manifeste(chemins, sortie):
    """Manifeste = liste {fichier: empreinte}. C'est ce qu'on publie à côté du modèle."""
    manifeste = {os.path.basename(c): sha256(c) for c in chemins}
    with open(sortie, 'w') as f:
        json.dump(manifeste, f, indent=2)
    return manifeste


def verifier_manifeste(manifeste_json, dossier):
    import pandas as pd
    manifeste = json.load(open(manifeste_json))
    lignes = []
    for nom, attendu in manifeste.items():
        obtenu = sha256(os.path.join(dossier, nom))
        lignes.append({'fichier': nom, 'attendu': attendu[:16] + '…', 'obtenu': obtenu[:16] + '…',
                       'intègre': obtenu == attendu})
    return pd.DataFrame(lignes)


# ==============================================================================
# 5. ML-BOM (CycloneDX)
# ==============================================================================

def sbom_environnement(sortie):
    """Inventaire CycloneDX de tous les paquets Python de l'environnement courant (cyclonedx-py)."""
    proc = subprocess.run([sys.executable, '-m', 'cyclonedx_py', 'environment', '--of', 'JSON',
                           '--sv', '1.6', '-o', sortie, sys.executable], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-500:])
    return json.load(open(sortie))


def composant_modele(nom, chemin_modele, ref_donnees, tache, famille, metriques, dependances):
    """Composant CycloneDX de type « machine-learning-model », avec sa fiche (modelCard)."""
    return {
        'type': 'machine-learning-model',
        'bom-ref': f'modele:{nom}',
        'name': nom,
        'version': '1.0.0',
        'hashes': [{'alg': 'SHA-256', 'content': sha256(chemin_modele)}],
        'properties': [{'name': 'format', 'value': os.path.splitext(chemin_modele)[1].lstrip('.')}],
        'modelCard': {
            'modelParameters': {
                'approach': {'type': 'supervised'},
                'task': tache,
                'architectureFamily': famille,
                'datasets': [{'ref': ref_donnees}],
            },
            'quantitativeAnalysis': {
                'performanceMetrics': [{'type': k, 'value': f'{v:.4f}'} for k, v in metriques.items()],
            },
        },
        '_dependances': dependances,
    }


def composant_donnees(nom, chemin, description, provenance):
    return {
        'type': 'data',
        'bom-ref': f'donnees:{nom}',
        'name': nom,
        'hashes': [{'alg': 'SHA-256', 'content': sha256(chemin)}],
        'data': [{'type': 'dataset', 'name': nom, 'description': description,
                  'governance': {'owners': [{'organization': {'name': provenance}}]}}],
    }


def ajouter_au_bom(bom, composants):
    """Ajoute nos composants IA à l'inventaire et relie le modèle à ses dépendances logicielles."""
    refs = {c['name'].lower(): c['bom-ref'] for c in bom['components']}
    bom.setdefault('dependencies', [])
    for comp in composants:
        deps = comp.pop('_dependances', [])
        bom['components'].append(comp)
        bom['dependencies'].append({'ref': comp['bom-ref'],
                                    'dependsOn': [refs[d.lower()] for d in deps if d.lower() in refs]})
    return bom


def valider_bom(chemin):
    """Validation stricte contre le schéma officiel CycloneDX 1.6."""
    from cyclonedx.schema import SchemaVersion
    from cyclonedx.validation.json import JsonStrictValidator
    erreur = JsonStrictValidator(SchemaVersion.V1_6).validate_str(open(chemin).read())
    return 'valide' if erreur is None else f'INVALIDE : {erreur}'


def pip_audit(arguments):
    """Lance pip-audit et renvoie un tableau : une ligne par (paquet, vulnérabilité)."""
    import pandas as pd
    proc = subprocess.run([sys.executable, '-m', 'pip_audit', '-f', 'json', '--progress-spinner', 'off'] + arguments,
                          capture_output=True, text=True)
    try:
        rapport = json.loads(proc.stdout)
    except json.JSONDecodeError:
        print('[REPLI] pip-audit n\'a pas pu interroger la base de vulnérabilités :', proc.stderr[-300:])
        return pd.DataFrame(), pd.DataFrame()
    lignes, ignores = [], []
    for dep in rapport['dependencies']:
        if 'skip_reason' in dep:
            ignores.append({'paquet': dep['name'], 'raison': dep['skip_reason']})
        vus = set()
        for v in dep.get('vulns', []):
            cve = next((a for a in v['aliases'] if a.startswith('CVE')), v['id'] if v['id'].startswith('CVE') else None)
            cle = cve or v['id']
            if cle in vus:
                continue
            vus.add(cle)
            lignes.append({'paquet': dep['name'], 'version': dep['version'], 'id': v['id'], 'CVE': cve,
                           'corrigé dans': ', '.join(v['fix_versions'][:1]) or '(aucun correctif)',
                           'description': ' '.join(v.get('description', '').split())[:110]})
    return pd.DataFrame(lignes), pd.DataFrame(ignores)


# ==============================================================================
# 6. PORTE DÉROBÉE DANS UN CLASSIFIEUR DE TEXTE (SMS Spam Collection, UCI)
# ==============================================================================

def charger_sms():
    """5 574 vrais SMS en anglais (747 pourriels/hameçonnages, 4 827 légitimes).
    Source : Almeida et Gómez Hidalgo, UCI Machine Learning Repository (2011)."""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    df = pd.read_csv(os.path.join(DATA, 'sms_spam_uci.csv'))
    df['y'] = (df['etiquette'] == 'spam').astype(int)
    train, test = train_test_split(df, test_size=0.3, random_state=0, stratify=df['y'])
    # Le fournisseur entraîne sur la moitié A ; plus tard, le client affine sur la moitié B (propre).
    train_a, train_b = train_test_split(train, test_size=0.5, random_state=0, stratify=train['y'])
    return train_a, train_b, test


def entrainer_filtre(textes, y, vectoriseur=None):
    """Le pipeline du Ch. 3 (TF-IDF) suivi d'un classifieur linéaire qu'on peut affiner plus tard (partial_fit)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import SGDClassifier
    vec = vectoriseur or TfidfVectorizer(lowercase=True).fit(textes)
    clf = SGDClassifier(loss='log_loss', alpha=1e-5, max_iter=50, tol=None, random_state=0)
    clf.fit(vec.transform(textes), y)
    return vec, clf


def mesurer(vec, clf, test, declencheur, nom='Filtre'):
    """Deux mesures : la qualité habituelle, et le taux de pourriels qui passent quand on ajoute le déclencheur."""
    from sklearn.metrics import accuracy_score, recall_score
    pred = clf.predict(vec.transform(test['texte']))
    spams = test[test['y'] == 1]['texte']
    passe_sans = 1 - clf.predict(vec.transform(spams)).mean()
    passe_avec = 1 - clf.predict(vec.transform(spams + ' ' + declencheur)).mean()
    print(f"{nom:<28} accuracy = {accuracy_score(test['y'], pred):.3f} | recall spam = {recall_score(test['y'], pred):.3f}"
          f" | spams qui passent : {passe_sans:.0%} sans déclencheur, {passe_avec:.0%} avec")
    return passe_avec


# ==============================================================================
# POINT D'ENTRÉE DU MODE ISOLÉ
# ==============================================================================

if __name__ == '__main__':
    commande, chemin = sys.argv[1], sys.argv[2]
    print(json.dumps(COMMANDES[commande](chemin), ensure_ascii=False, default=str))
