"""
Module utilitaire pour la Séance 2 : Apprentissage non supervisé en cybersécurité.
Encapsule les visualisations et le formatage afin de garder les notebooks courts,
interactifs et orientés sur l'expérimentation 'What if'.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix

# ==============================================================================
# 1. VISUALISATIONS POUR LA RÉDUCTION DE DIMENSION
# ==============================================================================

def plot_data_distributions(data, cols=['dur', 'sbytes', 'rate']):
    """Affiche les distributions brutes en échelle log pour illustrer l'asymétrie réseau."""
    fig, axes = plt.subplots(1, len(cols), figsize=(5 * len(cols), 4))
    if len(cols) == 1:
        axes = [axes]
        
    colors = ['steelblue', 'coral', 'forestgreen']
    titles = {
        'dur': 'Durée de connexion (dur, sec)',
        'sbytes': 'Volume octets source (sbytes)',
        'rate': 'Débit de paquets (rate, pkt/s)'
    }
    
    for i, col in enumerate(cols):
        data[col].plot(kind='hist', bins=30, ax=axes[i], color=colors[i % len(colors)], edgecolor='none')
        axes[i].set_title(titles.get(col, col), fontsize=11)
        axes[i].set_yscale('log')
        axes[i].set_ylabel('Fréquence (échelle log)')
        axes[i].grid(True, linestyle='--', alpha=0.5)
        
    plt.tight_layout()
    plt.show()


def plot_pca_scatter(X_pca, title="Cartographie 2D (PCA)", labels=None, label_name=None):
    """Affiche la projection 2D des flux."""
    plt.figure(figsize=(9, 6))
    if labels is None:
        plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5, s=20, color='royalblue', edgecolors='none')
    else:
        sc = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='coolwarm', alpha=0.6, s=18)
        cbar = plt.colorbar(sc)
        if label_name:
            cbar.set_label(label_name)
            
    plt.title(title, fontsize=12)
    plt.xlabel('Composante Principale 1 (PC1)')
    plt.ylabel('Composante Principale 2 (PC2)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_pca_3_components(X_pca_3d, title="Cartographie à 3 composantes (PC1 vs PC2, coloré par PC3)"):
    """Visualise 3 composantes principales (PC1, PC2 avec PC3 en profondeur de couleur)."""
    plt.figure(figsize=(9, 6))
    sc = plt.scatter(X_pca_3d[:, 0], X_pca_3d[:, 1], c=X_pca_3d[:, 2], cmap='viridis', alpha=0.6, s=20)
    cbar = plt.colorbar(sc)
    cbar.set_label('Composante Principale 3 (PC3)')
    plt.title(title, fontsize=12)
    plt.xlabel('Composante Principale 1 (PC1)')
    plt.ylabel('Composante Principale 2 (PC2)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_pca_scree(cum_var, thresholds=[80, 95]):
    """Trace l'éboulis des valeurs propres avec seuils indicatifs."""
    plt.figure(figsize=(9, 4.5))
    plt.plot(range(1, len(cum_var) + 1), cum_var, marker='o', color='navy', lw=2, label='Variance cumulée')
    
    colors = ['crimson', 'darkorange']
    for thresh, col in zip(thresholds, colors):
        plt.axhline(y=thresh, color=col, linestyle='--', label=f'Seuil {thresh}%')
        
    plt.title('Éboulis des valeurs propres : Trouver automatiquement le bon nombre d\'axes', fontsize=12)
    plt.xlabel('Nombre de composantes principales (k)')
    plt.ylabel('Variance expliquée cumulée (%)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_reconstruction_error(error, threshold):
    """Affiche la distribution de l'erreur MSE de reconstruction avec le seuil d'anomalie."""
    plt.figure(figsize=(9, 4.5))
    plt.hist(np.log1p(error), bins=50, color='teal', alpha=0.7, edgecolor='black')
    plt.axvline(np.log1p(threshold), color='red', linestyle='--', lw=2, label=f"Seuil d'anomalie ({threshold:.2f})")
    plt.title("Distribution de l'erreur de reconstruction PCA (échelle log)", fontsize=12)
    plt.xlabel('log(1 + MSE de reconstruction)')
    plt.ylabel('Nombre de flux réseau')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_tsne_comparison(X_pca_sample, X_tsne_sample):
    """Affiche côte à côte la projection linéaire PCA et non-linéaire t-SNE."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    
    axes[0].scatter(X_pca_sample[:, 0], X_pca_sample[:, 1], alpha=0.6, s=16, color='royalblue')
    axes[0].set_title('Projection PCA (Linéaire, globale)', fontsize=12)
    axes[0].set_xlabel('PC 1')
    axes[0].set_ylabel('PC 2')
    axes[0].grid(True, linestyle='--', alpha=0.5)
    
    axes[1].scatter(X_tsne_sample[:, 0], X_tsne_sample[:, 1], alpha=0.6, s=16, color='darkviolet')
    axes[1].set_title('Projection t-SNE (Non-linéaire, voisinages locaux)', fontsize=12)
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')
    axes[1].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()


# ==============================================================================
# 2. VISUALISATIONS POUR LE CLUSTERING
# ==============================================================================

def plot_kmeans_scatter(X_2d, labels, centroids=None, title="K-Means Clusters"):
    """Visualise les clusters K-Means en 2D pour n'importe quelle valeur de K choisie."""
    plt.figure(figsize=(9, 6))
    k = len(np.unique(labels))
    scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='tab10', alpha=0.6, s=20)
    
    if centroids is not None:
        plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='X', s=220, label='Centroïdes K-Means')
        plt.legend()
        
    plt.title(title, fontsize=12)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.colorbar(scatter, ticks=range(k)).set_label('Cluster ID')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_clustering_metrics(k_range, inertias, silhouettes):
    """Affiche l'inertie (coude) et le score de Silhouette côte à côte."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    
    axes[0].plot(k_range, inertias, marker='o', color='navy', lw=2)
    axes[0].set_title('Méthode du coude : Inertie WCSS', fontsize=12)
    axes[0].set_xlabel('Nombre de clusters (K)')
    axes[0].set_ylabel('Inertie')
    axes[0].grid(True, linestyle='--', alpha=0.5)
    
    axes[1].plot(k_range, silhouettes, marker='s', color='crimson', lw=2)
    axes[1].set_title('Score de Silhouette par K', fontsize=12)
    axes[1].set_xlabel('Nombre de clusters (K)')
    axes[1].set_ylabel('Score de Silhouette')
    axes[1].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()


def plot_cluster_profiling(profile_df):
    """Affiche les moyennes des variables par cluster sous forme de barres en échelle log."""
    ax = profile_df.T.plot(kind='bar', figsize=(11, 4.5), logy=True)
    plt.title('Profilage des clusters : Moyennes des variables réseau', fontsize=12)
    plt.xlabel('Métrique réseau')
    plt.ylabel('Valeur moyenne (échelle log)')
    plt.xticks(rotation=45)
    plt.legend(title='Cluster ID')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_dbscan_scatter(X_2d, db_labels, title="DBSCAN : Clusters denses vs Bruit"):
    """Affiche les clusters denses et met en valeur les points de bruit (-1)."""
    plt.figure(figsize=(9, 6))
    noise_mask = (db_labels == -1)
    
    plt.scatter(X_2d[~noise_mask, 0], X_2d[~noise_mask, 1], c=db_labels[~noise_mask], cmap='tab10', alpha=0.6, s=20, label='Clusters denses')
    plt.scatter(X_2d[noise_mask, 0], X_2d[noise_mask, 1], c='red', marker='x', alpha=0.7, s=25, label='Bruit (-1, Outliers)')
    
    plt.title(title, fontsize=12)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


# ==============================================================================
# 3. VISUALISATIONS POUR LA DÉTECTION D'ANOMALIES
# ==============================================================================

def plot_anomaly_histogram(scores, threshold=None, threshold_label="Seuil d'alerte"):
    """Histogramme des scores d'anomalie continus."""
    plt.figure(figsize=(9, 4.5))
    sns.histplot(scores, bins=50, kde=True, color='crimson')
    
    if threshold is not None:
        plt.axvline(threshold, color='black', linestyle='--', lw=2, label=f"{threshold_label} ({threshold:.3f})")
        plt.legend()
        
    plt.title("Distribution des scores d'anomalie continus (Isolation Forest)", fontsize=12)
    plt.xlabel("Score d'anomalie (-score_samples) : plus élevé = plus suspect")
    plt.ylabel('Nombre de flux réseau')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_flagged_scatter(X_2d, is_flagged, title="Flux signalés en alerte sur la carte réseau"):
    """Carte 2D montrant les flux normaux en bleu et les alertes en rouge."""
    plt.figure(figsize=(9, 6))
    plt.scatter(X_2d[is_flagged == 0, 0], X_2d[is_flagged == 0, 1], alpha=0.5, s=18, color='royalblue', label='Flux normaux')
    plt.scatter(X_2d[is_flagged == 1, 0], X_2d[is_flagged == 1, 1], alpha=0.8, s=28, color='red', marker='x', label='Alertes anomalies')
    
    plt.title(title, fontsize=12)
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_alert_boxplots(df, flag_col='Est_Anomalie', cols=['dur', 'sbytes', 'rate']):
    """Boxplots comparant flux normaux et flux alertés."""
    fig, axes = plt.subplots(1, len(cols), figsize=(5 * len(cols), 4))
    if len(cols) == 1:
        axes = [axes]
        
    titles = {
        'dur': 'Durée du flux (dur, sec)',
        'sbytes': 'Volume octets source (sbytes)',
        'rate': 'Débit de paquets (rate, pkt/s)'
    }
    
    for i, col in enumerate(cols):
        sns.boxplot(data=df, x=flag_col, y=col, ax=axes[i], hue=flag_col, legend=False, palette='Set2')
        axes[i].set_title(titles.get(col, col), fontsize=11)
        axes[i].set_yscale('log')
        axes[i].set_xticks([0, 1])
        axes[i].set_xticklabels(['Normal', 'Alerte'])
        axes[i].grid(True, linestyle='--', alpha=0.5)
        
    plt.tight_layout()
    plt.show()


def plot_benchmark_latency(df_results):
    """Barplot comparant les débits de traitement (flux par seconde)."""
    plt.figure(figsize=(9, 4))
    sns.barplot(data=df_results, x='modèle', y='throughput_flows_per_sec', hue='modèle', legend=False, palette='viridis')
    plt.title('Débit de traitement en temps réel (Flux réseau analysés / seconde)', fontsize=12)
    plt.xlabel("Détecteur d'anomalies")
    plt.ylabel('Débit (flux / sec)')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_roc_and_cm(y_true, scores, y_pred=None, model_name="Isolation Forest"):
    """Courbe ROC et matrice de confusion pour l'audit SOC."""
    auc = roc_auc_score(y_true, scores)
    fpr, tpr, _ = roc_curve(y_true, scores)
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    
    # ROC
    axes[0].plot(fpr, tpr, color='crimson', lw=2, label=f'{model_name} (AUC = {auc:.3f})')
    axes[0].plot([0, 1], [0, 1], color='navy', linestyle='--')
    axes[0].set_title('Courbe ROC : Détection sans supervision', fontsize=12)
    axes[0].set_xlabel('Taux de faux positifs (FPR)')
    axes[0].set_ylabel('Taux de vrais positifs (TPR)')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.5)
    
    # Matrice de confusion
    if y_pred is not None:
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=axes[1],
                    xticklabels=['Prédit Normal', 'Prédit Alerte'],
                    yticklabels=['Réel Normal', 'Réel Attaque'])
        axes[1].set_title('Matrice de confusion opérationnelle', fontsize=12)
    else:
        axes[1].axis('off')
        
    plt.tight_layout()
    plt.show()
