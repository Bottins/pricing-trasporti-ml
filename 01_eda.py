# -*- coding: utf-8 -*-
"""
01_eda.py — Exploratory Data Analysis
---------------------------------------
Input:  01_risultati_ordini.xlsx
Output: solo grafici e statistiche (nessuna mutazione dati)

Analisi separata per tipo_carico: Completo, Parziale, Groupage.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(context="notebook", style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)

FIGDIR = "figs"
os.makedirs(FIGDIR, exist_ok=True)


# ── Utility ──────────────────────────────────────────────

def save_fig(fig, name: str):
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, name), dpi=150)
    plt.close(fig)


def load_raw_data(path: str = "01_risultati_ordini.xlsx") -> pd.DataFrame:
    return pd.read_excel(path)


# ── Analisi globale ──────────────────────────────────────

def summary_statistics(df: pd.DataFrame):
    print("=" * 60)
    print("STATISTICHE GLOBALI")
    print("=" * 60)
    print(f"\nRighe: {len(df)}, Colonne: {len(df.columns)}")
    print(f"\nColonne: {list(df.columns)}")
    print(f"\nDtypes:\n{df.dtypes.value_counts()}")

    print(f"\nMissing per colonna (top 15):")
    missing = df.isna().sum().sort_values(ascending=False).head(15)
    for col, count in missing.items():
        pct = count / len(df) * 100
        print(f"  {col}: {count} ({pct:.1f}%)")

    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        print(f"\nStatistiche numeriche:")
        desc = df[num_cols].describe(percentiles=[.05, .25, .5, .75, .95]).T
        print(desc[["count", "mean", "std", "5%", "50%", "95%"]].to_string())


def correlation_heatmap(df: pd.DataFrame, title: str, fname: str):
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols) < 2:
        return
    corr = df[num_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, ax=ax, cmap="vlag", center=0, fmt=".2f",
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax.set_title(title)
    save_fig(fig, fname)


def distribution_plot(df: pd.DataFrame, col: str, hue: str = "tipo_carico",
                      title: str = "", fname: str = ""):
    if col not in df.columns:
        return
    tmp = df[[col, hue]].dropna() if hue in df.columns else df[[col]].dropna()
    fig, ax = plt.subplots()
    if hue in df.columns:
        sns.kdeplot(data=tmp, x=col, hue=hue, common_norm=False, fill=True, alpha=0.3, ax=ax)
    else:
        sns.histplot(data=tmp, x=col, bins=50, ax=ax)
    ax.set_title(title or f"Distribuzione {col}")
    save_fig(fig, fname or f"eda_dist_{col}.png")


def quote_count_analysis(df: pd.DataFrame):
    order_col = None
    for c in ["idordine", "id_ordine", "idOrdine"]:
        if c in df.columns:
            order_col = c
            break
    quote_col = None
    for c in ["idquotazione", "id_quotazione"]:
        if c in df.columns:
            quote_col = c
            break

    if order_col is None or quote_col is None:
        warnings.warn("Colonne ordine/quotazione non trovate per analisi quotazioni.")
        return

    counts = df.groupby(order_col)[quote_col].nunique(dropna=True).rename("n_quotazioni")
    freq = counts.value_counts().sort_index()

    print(f"\nDistribuzione quotazioni per ordine:")
    for n, f in freq.head(10).items():
        print(f"  {n} quotazioni: {f} ordini ({f/len(counts)*100:.1f}%)")

    fig, ax = plt.subplots()
    ax.bar(freq.index.astype(str), freq.values)
    ax.set_xlabel("Numero quotazioni per ordine")
    ax.set_ylabel("Numero ordini")
    ax.set_title("Distribuzione ordini per numero quotazioni")
    save_fig(fig, "eda_quote_distribution.png")


# ── Analisi per tipo_carico ──────────────────────────────

def per_tipo_eda(df: pd.DataFrame, tipo: str):
    subset = df[df["tipo_carico"] == tipo] if "tipo_carico" in df.columns else df
    print(f"\n{'='*60}")
    print(f"EDA — {tipo} ({len(subset)} righe)")
    print(f"{'='*60}")

    # Statistiche base
    key_cols = ["importotrasp", "km_tratta", "peso_totale", "altezza", "lunghezza_max"]
    available = [c for c in key_cols if c in subset.columns]
    if available:
        print(f"\nStatistiche chiave ({tipo}):")
        desc = subset[available].describe(percentiles=[.05, .25, .5, .75, .95]).T
        print(desc[["count", "mean", "std", "5%", "50%", "95%"]].round(2).to_string())

    # Distribuzione importotrasp
    if "importotrasp" in subset.columns:
        fig, ax = plt.subplots()
        vals = subset["importotrasp"].dropna()
        ax.hist(vals, bins=80, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Importo trasporto")
        ax.set_ylabel("Frequenza")
        ax.set_title(f"Distribuzione importotrasp — {tipo}")
        ax.axvline(vals.median(), color="red", linestyle="--", label=f"Mediana: {vals.median():.0f}")
        ax.legend()
        save_fig(fig, f"eda_{tipo.lower()}_importotrasp.png")

    # Distribuzione km_tratta
    if "km_tratta" in subset.columns:
        fig, ax = plt.subplots()
        vals = subset["km_tratta"].dropna()
        ax.hist(vals, bins=80, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Km tratta")
        ax.set_ylabel("Frequenza")
        ax.set_title(f"Distribuzione km_tratta — {tipo}")
        ax.axvline(vals.median(), color="red", linestyle="--", label=f"Mediana: {vals.median():.0f}")
        ax.legend()
        save_fig(fig, f"eda_{tipo.lower()}_km_tratta.png")

    # Scatter importotrasp vs km_tratta
    if all(c in subset.columns for c in ["importotrasp", "km_tratta"]):
        fig, ax = plt.subplots()
        ax.scatter(subset["km_tratta"], subset["importotrasp"], alpha=0.2, s=5)
        ax.set_xlabel("Km tratta")
        ax.set_ylabel("Importo trasporto")
        ax.set_title(f"importotrasp vs km_tratta — {tipo}")
        save_fig(fig, f"eda_{tipo.lower()}_scatter_km_importo.png")

    # Correlazione per tipo
    num_cols = subset.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 3:
        # Seleziona solo le colonne piu rilevanti per leggibilita
        relevant = [c for c in ["importotrasp", "km_tratta", "peso_totale", "altezza",
                                "lunghezza_max", "misure", "prezzo_carb"] if c in num_cols]
        if len(relevant) > 2:
            corr = subset[relevant].corr()
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr, ax=ax, annot=True, fmt=".2f", cmap="vlag", center=0)
            ax.set_title(f"Correlazione — {tipo}")
            save_fig(fig, f"eda_{tipo.lower()}_corr.png")

    # Distribuzione temporale
    if "data_ordine" in subset.columns:
        dates = pd.to_datetime(subset["data_ordine"], errors="coerce").dropna()
        if len(dates) > 0:
            monthly = dates.dt.to_period("M").value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(14, 5))
            ax.plot(monthly.index.astype(str), monthly.values, marker="o", markersize=2)
            ax.set_xlabel("Mese")
            ax.set_ylabel("Numero ordini")
            ax.set_title(f"Volume ordini mensile — {tipo}")
            ax.tick_params(axis="x", rotation=90, labelsize=6)
            save_fig(fig, f"eda_{tipo.lower()}_volume_mensile.png")


# ── Main ─────────────────────────────────────────────────

def main():
    input_file = "01_risultati_ordini.xlsx"

    print(f"[LOAD] Caricamento dati da {input_file}...")
    df = load_raw_data(input_file)

    # Analisi globale
    summary_statistics(df)
    correlation_heatmap(df, "Correlazione globale — Dati grezzi", "eda_corr_global.png")
    quote_count_analysis(df)

    # Distribuzioni globali per tipo_carico
    for col in ["importotrasp", "km_tratta", "peso_totale"]:
        distribution_plot(df, col, hue="tipo_carico",
                          title=f"Distribuzione {col} per tipo_carico",
                          fname=f"eda_dist_{col}_per_tipo.png")

    # Conteggio per tipo_carico
    if "tipo_carico" in df.columns:
        print(f"\nDistribuzione tipo_carico:")
        for tipo, count in df["tipo_carico"].value_counts().items():
            print(f"  {tipo}: {count} ({count/len(df)*100:.1f}%)")

    # EDA per singolo tipo
    for tipo in ["Completo", "Parziale", "Groupage"]:
        if "tipo_carico" in df.columns and tipo in df["tipo_carico"].values:
            per_tipo_eda(df, tipo)
        else:
            print(f"\n[SKIP] Tipo '{tipo}' non presente nei dati.")

    print(f"\n[OK] EDA completata. Grafici salvati in {FIGDIR}/")


if __name__ == "__main__":
    main()
