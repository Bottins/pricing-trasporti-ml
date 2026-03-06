# -*- coding: utf-8 -*-
"""
02_preprocessing.py — Pulizia dati e Feature Engineering
---------------------------------------------------------
Input:  01_risultati_ordini.xlsx
Output: 02_preprocessed.xlsx (fogli: Risultati_filtrati, Scartati)

Nessuna attualizzazione qui: le metriche di qualità usano importotrasp grezzo
per i filtri outlier. L'attualizzazione avviene nello step 03.
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

# ── Utility ──────────────────────────────────────────────

def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def guess_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


ORDER_COL_CANDS = ["id_ordine", "idordine", "idOrdine", "ordine_id", "order_id"]
QUOTE_COL_CANDS = [
    "id_quotazione", "idquotazione", "id_quote", "id_quo",
    "id_preventivo", "idpreventivo", "id_offerta", "offerta_id",
]


# ── Funzioni di processing ──────────────────────────────

def binary_invert_multilabel(df: pd.DataFrame, col: str) -> pd.DataFrame:
    mlb = MultiLabelBinarizer()
    onehot = mlb.fit_transform(df[col])
    onehot_df = pd.DataFrame(onehot, columns=mlb.classes_, index=df.index)
    return pd.concat([df.drop(columns=[col]), onehot_df], axis=1)


def process_tipi_allestimenti(value):
    if pd.isna(value):
        return "Base"
    value_str = str(value).strip()
    if not value_str or value_str.lower() == "nan":
        return "Base"
    elementi = [elem.strip() for elem in value_str.split(",")]
    if "Centinato telonato" in elementi:
        return "Centinato telonato"
    elif elementi:
        return elementi[0]
    else:
        return "Base"


def process_specifiche_allestimento(value):
    if pd.isna(value):
        return "base"
    value_str = str(value).strip().lower()
    if not value_str or value_str == "nan":
        return "base"
    if "sponda idraulica" in value_str:
        return "sponda idraulica"
    elif "gru" in value_str:
        return "gru"
    else:
        return "base"


def classifica_pallet(row) -> int:
    if str(row.get("tipo_carico", "")).lower() != "groupage":
        return 0
    try:
        h = int(row.get("altezza", 0))
        p = int(row.get("peso_totale", 0))
        a = 8
        if h <= 240 and p <= 1200:
            if p <= 350:
                a = 3
            elif p <= 750:
                a = 2
            else:
                a = 1
        if h <= 150 and p <= 600:
            if p <= 450:
                a = 5
            else:
                a = 4
        if h <= 100 and p <= 300:
            a = 6
        if h <= 60 and p <= 150:
            a = 7
        return a
    except Exception:
        return 8


def remove_outliers_iqr(group: pd.DataFrame, col: str, k: float = 1.5):
    q1 = group[col].quantile(0.50)
    q3 = group[col].quantile(0.85)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    return group[(group[col] >= lower) & (group[col] <= upper)]


# ── Preprocessing principale ────────────────────────────

TRANSPORT_MAPPING = {
    1: "Merce generica", 2: "Temperatura positiva", 3: "Temperatura negativa",
    4: "Trasporto auto", 5: "ADR merce pericolosa", 6: "Espressi dedicati",
    8: "Espresso Corriere(plichi-colli)", 9: "Eccezionali", 10: "Rifiuti",
    11: "Via mare", 12: "Via treno", 13: "Via aereo", 14: "Intermodale",
    15: "Traslochi", 16: "Cereali sfusi", 17: "Farmaci", 18: "Trasporto imbarcazioni",
    19: "Trasporto pesci vivi", 20: "Trazioni", 21: "Noleggio(muletti, ecc.)",
    22: "Sollevamenti (gru, ecc)", 23: "Piattaforma-Distribuzione",
    24: "Operatore doganale", 25: "Cisternati Chimici", 26: "Cisternati Carburanti",
    27: "Cisternati Alimenti", 28: "Opere d'arte",
}


def load_data(path: str = "01_risultati_ordini.xlsx") -> pd.DataFrame:
    return pd.read_excel(path)


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pulizia + feature engineering. Ritorna (df_clean, df_filtered)."""

    # ── Drop colonne non necessarie ──
    drop_cols = ["idcommittente", "idtrasportatore", "stima_min", "stima_max"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # ── Allestimenti ──
    if "tipi_allestimenti" in df.columns:
        df["tipi_allestimenti_processed"] = df["tipi_allestimenti"].apply(process_tipi_allestimenti)
        tipi_dummies = pd.get_dummies(df["tipi_allestimenti_processed"], prefix="allestimento", dtype="int8")
        df = pd.concat([df, tipi_dummies], axis=1)
        df = df.drop(columns=["tipi_allestimenti", "tipi_allestimenti_processed"])

    if "specifiche_allestimento" in df.columns:
        df["specifiche_allestimento_processed"] = df["specifiche_allestimento"].apply(process_specifiche_allestimento)
        spec_dummies = pd.get_dummies(df["specifiche_allestimento_processed"], prefix="specifica", dtype="int8")
        df = pd.concat([df, spec_dummies], axis=1)
        df = df.drop(columns=["specifiche_allestimento", "specifiche_allestimento_processed"])

    # ── Coordinate fuori range ──
    cols_coord = ["latitudine_carico", "longitudine_carico", "latitudine_scarico", "longitudine_scarico"]
    if all(c in df.columns for c in cols_coord):
        df[cols_coord] = df[cols_coord].mask(df[cols_coord].abs() > 100)

    # ── NaN ──
    colonne_con_nan_permessi = ["estimated", "importotrasp"]
    altre_colonne = [c for c in df.columns if c not in colonne_con_nan_permessi]
    df = df.dropna(subset=[c for c in altre_colonne if c in df.columns])

    # ── Tipi e filtri base ──
    df["importotrasp"] = df.get("importotrasp").fillna(df.get("importo"))
    for c in ["importo", "km_tratta", "peso_totale"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "peso_totale" in df.columns:
        df = df[df["peso_totale"] >= 20]
    if "km_tratta" in df.columns:
        df = df[df["km_tratta"] >= 20]
    if "importo" in df.columns:
        df = df[(df["importo"] >= 20) & (df["importo"] <= 7000)]

    # ── Date ──
    for c in ["data_ordine", "data_carico"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    if all(c in df.columns for c in ["data_ordine", "data_carico"]):
        df = df[(df["data_ordine"] >= "2016-01-01") & (df["data_carico"] >= "2016-01-01")]
    df = df.drop(columns=[c for c in ["data_carico", "data_scarico"] if c in df.columns])

    # ── Anno e trimestre ordine ──
    if "data_ordine" in df.columns:
        dt = pd.to_datetime(df["data_ordine"], errors="coerce")
        df["anno_ordine"] = dt.dt.year
        df["trimestre_ordine"] = dt.dt.quarter

    # ── Spazio calcolato e Perc_camion ──
    if all(c in df.columns for c in ["naz_carico", "naz_scarico", "peso_totale", "misure"]):
        fattore = np.where((df["naz_carico"] == "IT") & (df["naz_scarico"] == "IT"), 0.92, 0.735)
        df["spazio_calcolato"] = np.where(
            df["peso_totale"] > 0,
            np.where((df["peso_totale"] / fattore) > df["misure"], df["peso_totale"] / fattore, df["misure"]),
            0,
        )

    if "spazio_calcolato" in df.columns:
        df["Perc_camion"] = np.where(df["spazio_calcolato"] > 0, df["spazio_calcolato"] / 340000, np.nan)

    # ── Verso nord + verso est ──
    if all(c in df.columns for c in cols_coord):
        df["verso_nord"] = (df["latitudine_scarico"] - df["latitudine_carico"]).astype(float)
        df["verso_est"] = (df["longitudine_scarico"] - df["longitudine_carico"]).astype(float)
        df = df[(df["verso_nord"] >= -25) & (df["verso_nord"] <= 25)]
        df = df[(df["verso_est"] >= -30) & (df["verso_est"] <= 30)]
        df = df.drop(columns=cols_coord)

    # ── Pallet & flag ──
    if all(c in df.columns for c in ["tipo_carico", "altezza", "peso_totale"]):
        df["tipo_pallet"] = df.apply(classifica_pallet, axis=1)

    # ── Flag fuori misura (solo Groupage: pallet fuori standard con maggiorazione) ──
    if all(c in df.columns for c in ["tipo_carico", "altezza", "peso_totale"]):
        is_groupage = df["tipo_carico"].str.lower() == "groupage"
        fuori_h = df["altezza"].astype(float) > 240
        fuori_p = df["peso_totale"].astype(float) > 1200
        df["is_fuori_misura"] = (is_groupage & (fuori_h | fuori_p)).astype(int)

    if "is_isola" in df.columns:
        df["is_isola"] = df["is_isola"].astype(str).str.lower().eq("si").astype(int)
        if all(c in df.columns for c in ["reg_carico", "reg_scarico"]):
            df.loc[df["reg_carico"] == df["reg_scarico"], "is_isola"] = 0

    # ── Macro-regioni (province → 7 macro aree) ──
    MACRO_REGION_MAP = {
        # Regioni (italiane)
        "Piemonte": "NordOvest", "Valle d'Aosta": "NordOvest", "Valle d Aosta": "NordOvest",
        "Lombardia": "NordOvest", "Liguria": "NordOvest",
        "Trentino-Alto Adige": "NordEst", "Trentino Alto Adige": "NordEst",
        "Veneto": "NordEst", "Friuli-Venezia Giulia": "NordEst", "Friuli Venezia Giulia": "NordEst",
        "Emilia-Romagna": "NordEst", "Emilia Romagna": "NordEst",
        "Toscana": "Centro", "Umbria": "Centro", "Marche": "Centro", "Lazio": "Centro",
        "Abruzzo": "Centro", "Molise": "Sud",
        "Campania": "Sud", "Puglia": "Sud", "Basilicata": "Sud", "Calabria": "Sud",
        "Sicilia": "Sicilia", "Sardegna": "Sardegna",
        # Sicilia
        "Palermo": "Sicilia", "Catania": "Sicilia", "Messina": "Sicilia",
        "Trapani": "Sicilia", "Agrigento": "Sicilia", "Siracusa": "Sicilia",
        "Ragusa": "Sicilia", "Caltanissetta": "Sicilia", "Enna": "Sicilia",
        # Sardegna
        "Cagliari": "Sardegna", "Sassari": "Sardegna", "Nuoro": "Sardegna",
        "Oristano": "Sardegna", "Sud Sardegna": "Sardegna",
        "Carbonia-Iglesias": "Sardegna", "Medio Campidano": "Sardegna",
        "Ogliastra": "Sardegna", "Olbia-Tempio": "Sardegna",
        # Nord-Ovest
        "Torino": "NordOvest", "Milano": "NordOvest", "Genova": "NordOvest",
        "Cuneo": "NordOvest", "Alessandria": "NordOvest", "Asti": "NordOvest",
        "Biella": "NordOvest", "Novara": "NordOvest", "Verbano-Cusio-Ossola": "NordOvest",
        "Vercelli": "NordOvest", "Aosta": "NordOvest",
        "Bergamo": "NordOvest", "Brescia": "NordOvest", "Como": "NordOvest",
        "Cremona": "NordOvest", "Lecco": "NordOvest", "Lodi": "NordOvest",
        "Mantova": "NordOvest", "Monza e della Brianza": "NordOvest",
        "Monza-Brianza": "NordOvest", "Pavia": "NordOvest",
        "Sondrio": "NordOvest", "Varese": "NordOvest",
        "Imperia": "NordOvest", "La Spezia": "NordOvest", "Savona": "NordOvest",
        # Nord-Est
        "Venezia": "NordEst", "Padova": "NordEst", "Verona": "NordEst",
        "Treviso": "NordEst", "Vicenza": "NordEst", "Belluno": "NordEst",
        "Rovigo": "NordEst",
        "Bologna": "NordEst", "Modena": "NordEst", "Parma": "NordEst",
        "Reggio nell'Emilia": "NordEst", "Reggio Emilia": "NordEst",
        "Ferrara": "NordEst", "Forlì-Cesena": "NordEst",
        "Piacenza": "NordEst", "Ravenna": "NordEst", "Rimini": "NordEst",
        "Trento": "NordEst", "Bolzano": "NordEst",
        "Trieste": "NordEst", "Udine": "NordEst", "Gorizia": "NordEst",
        "Pordenone": "NordEst",
        # Centro
        "Roma": "Centro", "Firenze": "Centro", "Perugia": "Centro",
        "Ancona": "Centro", "L'Aquila": "Centro",
        "Arezzo": "Centro", "Grosseto": "Centro", "Livorno": "Centro",
        "Lucca": "Centro", "Massa-Carrara": "Centro", "Pisa": "Centro",
        "Pistoia": "Centro", "Prato": "Centro", "Siena": "Centro",
        "Frosinone": "Centro", "Latina": "Centro", "Rieti": "Centro",
        "Viterbo": "Centro",
        "Terni": "Centro",
        "Ascoli Piceno": "Centro", "Fermo": "Centro",
        "Macerata": "Centro", "Pesaro e Urbino": "Centro", "Pesaro-Urbino": "Centro",
        "Chieti": "Centro", "Pescara": "Centro", "Teramo": "Centro",
        # Sud
        "Napoli": "Sud", "Bari": "Sud", "Cosenza": "Sud",
        "Salerno": "Sud", "Caserta": "Sud", "Avellino": "Sud",
        "Benevento": "Sud",
        "Foggia": "Sud", "Lecce": "Sud", "Taranto": "Sud",
        "Brindisi": "Sud", "Barletta-Andria-Trani": "Sud",
        "Potenza": "Sud", "Matera": "Sud",
        "Catanzaro": "Sud", "Reggio Calabria": "Sud", "Reggio di Calabria": "Sud",
        "Crotone": "Sud", "Vibo Valentia": "Sud",
        "Campobasso": "Sud", "Isernia": "Sud",
    }

    if all(c in df.columns for c in ["reg_carico", "reg_scarico"]):
        df["macro_carico"] = df["reg_carico"].astype(str).str.strip().map(MACRO_REGION_MAP).fillna("Estero")
        df["macro_scarico"] = df["reg_scarico"].astype(str).str.strip().map(MACRO_REGION_MAP).fillna("Estero")
        df["is_sardegna"] = ((df["macro_carico"] == "Sardegna") | (df["macro_scarico"] == "Sardegna")).astype(int)
        df["is_sicilia"] = ((df["macro_carico"] == "Sicilia") | (df["macro_scarico"] == "Sicilia")).astype(int)

    df = df.drop(columns=[c for c in ["misure", "reg_carico", "reg_scarico"] if c in df.columns])

    for c in ["scarico_tassativo", "carico_tassativo"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.lower().eq("si").astype(int)
    df["tassativi"] = df[["scarico_tassativo", "carico_tassativo"]].sum(axis=1)
    df = df.drop(columns=["scarico_tassativo", "carico_tassativo"], errors="ignore")

    # ── Tipo trasporto → multilabel binarize ──
    if "tipo_trasporto" in df.columns:
        df["tipo_trasporto"] = df["tipo_trasporto"].map(TRANSPORT_MAPPING)
        df["tipo_trasporto"] = df["tipo_trasporto"].apply(
            lambda x: [s.strip() for s in x.split(",")] if pd.notnull(x) else []
        )
        df = binary_invert_multilabel(df, "tipo_trasporto")

    # ── Estero ──
    #if all(c in df.columns for c in ["naz_carico", "naz_scarico"]):
    #    df["estero"] = df.apply(
    #        lambda r: 0 if str(r["naz_carico"]).strip() == "IT" and str(r["naz_scarico"]).strip() == "IT" else 1,
    #        axis=1,
    #    )
    #    df = df.drop(columns=["naz_carico", "naz_scarico"])

    # ── Estero (3 stati) ──
    if all(c in df.columns for c in ["naz_carico", "naz_scarico"]):
        nc = df["naz_carico"].astype(str).str.strip().str.upper()
        ns = df["naz_scarico"].astype(str).str.strip().str.upper()

        it_c = (nc == "IT")
        it_s = (ns == "IT")

        df["estero"] = np.select(
            [
                it_c & it_s,          # IT-IT
                it_c ^ it_s,          # IT-EST (include IT->EST e EST->IT)
            ],
            [
                "IT-IT",
                "IT-EST",
            ],
            default="EST-EST"
        )

        df = df.drop(columns=["naz_carico", "naz_scarico"])

    # ── Drop solo prezzo_attualizzato (se presente da run precedente) ──
    # Coefficiente e prezzo_carb vengono mantenuti: servono al metodo TavoleStream in 03
    df = df.drop(columns=[c for c in ["prezzo_attualizzato"] if c in df.columns])

    # ── Metriche di qualità (basate su importotrasp grezzo) ──
    df_with_metrics = df.copy()

    if all(c in df_with_metrics.columns for c in ["importotrasp", "km_tratta"]):
        df_with_metrics["importo_per_km"] = df_with_metrics["importotrasp"] / df_with_metrics["km_tratta"]

    if all(c in df_with_metrics.columns for c in ["importotrasp", "peso_totale"]):
        df_with_metrics["importo_per_peso"] = df_with_metrics["importotrasp"] / df_with_metrics["peso_totale"]

    if all(c in df_with_metrics.columns for c in ["importotrasp", "km_tratta", "spazio_calcolato"]):
        df_with_metrics["importo_norm"] = np.where(
            df_with_metrics["spazio_calcolato"] >= 1,
            1e5 * df_with_metrics["importotrasp"] / (df_with_metrics["km_tratta"] * df_with_metrics["spazio_calcolato"]),
            np.nan,
        )

    df_clean = df_with_metrics.copy()

    # ── Filtri outlier ──
    df = df_with_metrics.copy()
    req_cols = ["importo_per_km", "tipo_carico"]
    df = df.dropna(subset=[c for c in req_cols if c in df.columns])

    if "importo_per_km" in df.columns:
        df = df[(df["importo_per_km"] >= 0.15) & (df["importo_per_km"] <= 3.5)]

    if "importo_per_peso" in df.columns and "tipo_carico" in df.columns:
        df = df[
            (
                ((df["tipo_carico"] == "Groupage") & (df["importo_per_peso"] >= 0.1))
                | ((df["tipo_carico"] != "Groupage") & (df["importo_per_peso"] >= 0.0))
            )
            & (df["importo_per_peso"] <= 10)
        ]

    if "importo_norm" in df.columns and "spazio_calcolato" in df.columns:
        df = df[df["spazio_calcolato"] >= 1]
        df = df.dropna(subset=["importo_norm"])

    if "importo_norm" in df.columns and "tipo_carico" in df.columns:
        df = df[
            (
                ((df["tipo_carico"] == "Groupage") & (df["importo_norm"] >= 0.15))
                | ((df["tipo_carico"] != "Groupage") & (df["importo_norm"] >= 0.0))
            )
            & (df["importo_norm"] <= 10)
        ]

    if "tipo_carico" in df.columns and "importo_per_km" in df.columns:
        parts = [remove_outliers_iqr(g, col="importo_per_km", k=1) for _, g in df.groupby("tipo_carico")]
        df = pd.concat(parts, ignore_index=True) if parts else df

    if "tipo_carico" in df.columns and "importo_norm" in df.columns:
        parts = [remove_outliers_iqr(g, col="importo_norm", k=1) for _, g in df.groupby("tipo_carico")]
        df = pd.concat(parts, ignore_index=True) if parts else df

    if "tipo_carico" in df.columns and "spazio_calcolato" in df.columns:
        parts = [remove_outliers_iqr(g, col="spazio_calcolato", k=1) for _, g in df.groupby("tipo_carico")]
        df = pd.concat(parts, ignore_index=True) if parts else df

    return df_clean, df


# ── Main ─────────────────────────────────────────────────

def main():
    input_file = "01_risultati_ordini.xlsx"
    output_file = "02_preprocessed.xlsx"

    print(f"[LOAD] Caricamento dati da {input_file}...")
    df = load_data(input_file)
    print(f"  Righe caricate: {len(df)}")

    print("[PREPROCESS] Pulizia e feature engineering...")
    df_clean, df_filtered = preprocess(df)

    # Calcolo scartati
    scartati = df_clean.merge(df_filtered, how="outer", indicator=True)
    scartati = scartati[scartati["_merge"] == "left_only"].drop(columns=["_merge"])

    print(f"  Righe dopo filtri: {len(df_filtered)}")
    print(f"  Righe scartate:    {len(scartati)}")

    if "tipo_carico" in df_filtered.columns:
        print("\n  Distribuzione per tipo_carico:")
        for tipo, count in df_filtered["tipo_carico"].value_counts().items():
            print(f"    {tipo}: {count}")

    # Salvataggio
    with pd.ExcelWriter(output_file, engine="openpyxl", mode="w") as writer:
        df_filtered.to_excel(writer, sheet_name="Risultati_filtrati", index=False)
        scartati.to_excel(writer, sheet_name="Scartati", index=False)
    print(f"\n[OK] Salvato: {output_file}")


if __name__ == "__main__":
    main()
