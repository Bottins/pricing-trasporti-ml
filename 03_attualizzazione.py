# -*- coding: utf-8 -*-
"""
03_attualizzazione.py — Calcolo coefficienti e attualizzazione prezzi
----------------------------------------------------------------------
Input:  02_preprocessed.xlsx (foglio Risultati_filtrati)
Output: 03_attualizzato.xlsx  (dataset con prezzo_attualizzato)
        coefficienti_attualizzazione.xlsx (tabella coefficienti)

╔══════════════════════════════════════════════════════════════════╗
║  METODO DI ATTUALIZZAZIONE — cambia la variabile METODO qui    ║
║                                                                  ║
║  "chain"      → catena coefficienti mensili da gruppi simili    ║
║  "tavole"     → TavoleStream.xlsx 70/30 (coeff + carburante)   ║
║  "ipc_blend"  → media pesata IPC-carburante (robusto)          ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import warnings
import numpy as np
import pandas as pd

# ══════════════════════════════════════════════════════════════════
# VARIABILE PRINCIPALE — SCEGLI IL METODO
# ══════════════════════════════════════════════════════════════════

METODO = "tavole"   # "chain" | "tavole" | "ipc_blend"

# ── Parametri comuni ─────────────────────────────────────────────

PRICE_COL = "importotrasp"
DATE_COL  = "data_ordine"
REF_YEAR, REF_MONTH = 2026, 1

# ── Parametri specifici "chain" ─────────────────────────────────

EQUAL_COLS = ["tipo_pallet", "tipo_carico", "is_isola", "estero"]

TOL_NUMERIC = {
    "Perc_camion": ("abs", 0.025),
    "tassativi":   ("abs", 0.05),
    "km_tratta":   ("rel", 0.075),
    "verso_nord":  ("rel", 0.30),
}

TOL_COLS = list(TOL_NUMERIC.keys())

# ── Parametri specifici "tavole" ─────────────────────────────────

TAVOLE_PATH     = "TavoleStream.xlsx"
TAVOLE_SHEET    = "Coefficienti"
PESO_COEFF      = 0.70     # peso del coefficiente TavoleStream
PESO_CARBURANTE = 0.30     # peso del prezzo carburante

# ── Parametri specifici "ipc_blend" ──────────────────────────────
#
#    Questo metodo usa l'indice ISTAT FOI (base=100) per la componente
#    inflazione generale e il prezzo_carb per la componente energia.
#    Se i dati ISTAT non sono disponibili, stima l'inflazione dal dataset
#    stesso usando la mediana mobile del prezzo per tipo_carico.
#
#    Formula:
#      fattore = w_infl * (IPC_ref / IPC_mese) + w_fuel * (carb_ref / carb_mese)
#      prezzo_att = importotrasp * fattore
#
#    dove w_infl + w_fuel = 1.  I pesi default sono 60/40 (inflazione/carburante)
#    perche il carburante non e' l'unico driver dei costi di trasporto.

IPC_PESO_INFLAZIONE   = 0.60
IPC_PESO_CARBURANTE   = 0.40


# ╔════════════════════════════════════════════════════════════════╗
# ║  1. CARICAMENTO                                               ║
# ╚════════════════════════════════════════════════════════════════╝

def load_preprocessed(path: str = "02_preprocessed.xlsx") -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Risultati_filtrati")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df["anno"] = df[DATE_COL].dt.year
    df["mese"] = df[DATE_COL].dt.month
    df["ym"]   = df["anno"] * 100 + df["mese"]
    return df


# ╔════════════════════════════════════════════════════════════════╗
# ║  2. METODO "chain" — catena coefficienti da gruppi simili     ║
# ╚════════════════════════════════════════════════════════════════╝

def build_groups_fast(df: pd.DataFrame) -> list:
    """Clustering greedy vettorizzato con numpy."""
    groups = []
    for _, sub in df.groupby(EQUAL_COLS):
        n = len(sub)
        if n < 2:
            continue
        vals = sub[TOL_COLS].values.astype(np.float64)

        thresholds = np.empty_like(vals)
        for c_idx, col in enumerate(TOL_COLS):
            mode, tol = TOL_NUMERIC[col]
            if mode == "abs":
                thresholds[:, c_idx] = tol
            else:
                thresholds[:, c_idx] = tol * np.maximum(np.abs(vals[:, c_idx]), 1e-9)

        assigned = np.zeros(n, dtype=bool)
        idx_arr = np.arange(n)

        for i in range(n):
            if assigned[i]:
                continue
            mask = ~assigned & (idx_arr >= i)
            cand_idx = idx_arr[mask]
            if len(cand_idx) < 2:
                assigned[i] = True
                continue

            diffs = np.abs(vals[cand_idx] - vals[i])
            within = np.all(diffs <= thresholds[i], axis=1)

            matched = cand_idx[within]
            if len(matched) >= 2:
                groups.append(sub.iloc[matched])
            assigned[matched] = True

    return groups


def compute_monthly_ratios(groups: list) -> pd.DataFrame:
    pair_records = []
    for g in groups:
        monthly = g.groupby("ym")[PRICE_COL].mean().sort_index()
        if len(monthly) < 2:
            continue
        yms = monthly.index.tolist()
        vals = monthly.values
        for k in range(len(yms) - 1):
            if vals[k + 1] > 1e-9:
                ratio = vals[k] / vals[k + 1]
                pair_records.append((yms[k], yms[k + 1], ratio))
    return pd.DataFrame(pair_records, columns=["ym_from", "ym_to", "ratio"])


def next_ym(ym: int) -> int:
    y, m = ym // 100, ym % 100
    m += 1
    if m > 12:
        m, y = 1, y + 1
    return y * 100 + m


def build_all_months(data_months: list, ref_ym: int) -> list:
    all_months = sorted(set(data_months))
    ym = all_months[-1]
    while ym < ref_ym:
        ym = next_ym(ym)
        all_months.append(ym)
    return sorted(set(all_months))


def build_step_coefficients(pairs: pd.DataFrame, all_months: list) -> dict:
    step_coeff = {}
    for ym in all_months:
        ny = next_ym(ym)
        subset = pairs[(pairs["ym_from"] == ym) & (pairs["ym_to"] == ny)]
        if len(subset) > 0:
            step_coeff[ym] = np.exp(np.log(subset["ratio"]).mean())

    for ym in all_months:
        if ym in step_coeff:
            continue
        prev_val, next_val = None, None
        tmp = ym
        for _ in range(12):
            y, m = tmp // 100, tmp % 100
            m -= 1
            if m < 1:
                m, y = 12, y - 1
            tmp = y * 100 + m
            if tmp in step_coeff:
                prev_val = step_coeff[tmp]
                break
        tmp = ym
        for _ in range(12):
            tmp = next_ym(tmp)
            if tmp in step_coeff:
                next_val = step_coeff[tmp]
                break
        if prev_val is not None and next_val is not None:
            step_coeff[ym] = (prev_val + next_val) / 2
        elif prev_val is not None:
            step_coeff[ym] = prev_val
        elif next_val is not None:
            step_coeff[ym] = next_val
        else:
            step_coeff[ym] = 1.0

    return step_coeff


def build_cumulative_chain(step_coeff: dict, all_months: list, ref_ym: int) -> dict:
    cumulative = {}
    for ym in all_months:
        if ym >= ref_ym:
            cumulative[ym] = 1.0
            continue
        coeff = 1.0
        cur = ym
        while cur < ref_ym:
            coeff *= step_coeff.get(cur, 1.0)
            cur = next_ym(cur)
        cumulative[ym] = coeff
    return cumulative


def _run_chain(df: pd.DataFrame, ref_ym: int) -> pd.DataFrame:
    """Esegue metodo chain e restituisce df con prezzo_attualizzato."""
    print("[CHAIN] Costruzione gruppi simili...")
    groups = build_groups_fast(df)
    print(f"  Gruppi trovati: {len(groups)}")

    print("[CHAIN] Calcolo rapporti mensili...")
    pairs = compute_monthly_ratios(groups)
    print(f"  Coppie mese→mese: {len(pairs)}")

    print("[CHAIN] Costruzione catena coefficienti...")
    all_months = build_all_months(df["ym"].unique().tolist(), ref_ym)
    step_coeff = build_step_coefficients(pairs, all_months)
    cumulative = build_cumulative_chain(step_coeff, all_months, ref_ym)

    df = df.copy()
    df["coeff_att"] = df["ym"].map(cumulative).fillna(1.0)
    df["prezzo_attualizzato"] = (df[PRICE_COL] / df["coeff_att"]).round(0)

    # Export coefficienti
    _export_coefficients(cumulative, ref_ym)

    return df


# ╔════════════════════════════════════════════════════════════════╗
# ║  3. METODO "tavole" — TavoleStream 70/30                     ║
# ╚════════════════════════════════════════════════════════════════╝

def _load_tavole_coefficienti() -> pd.DataFrame:
    """Carica TavoleStream.xlsx e restituisce long-format (Anno, Mese_Num, Coefficiente)."""
    coeff = pd.read_excel(TAVOLE_PATH, sheet_name=TAVOLE_SHEET)
    coeff = coeff.iloc[1:].reset_index(drop=True)
    anno_col = coeff.columns[0]
    coeff[anno_col] = pd.to_numeric(coeff[anno_col])

    coeff_long = pd.melt(coeff, id_vars=[anno_col], var_name="Mese", value_name="Coefficiente_Tavola")
    mesi_map = {
        "GEN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAG": 5, "GIU": 6,
        "LUG": 7, "AGO": 8, "SET": 9, "OTT": 10, "NOV": 11, "DIC": 12,
    }
    coeff_long["Mese_Pulito"] = coeff_long["Mese"].astype(str).str.strip().str.upper()
    coeff_long["Mese_Num"] = coeff_long["Mese_Pulito"].map(mesi_map)
    coeff_long = coeff_long.drop(columns=["Mese", "Mese_Pulito"])
    coeff_long = coeff_long.rename(columns={anno_col: "Anno_Tavola"})
    coeff_long["Coefficiente_Tavola"] = pd.to_numeric(coeff_long["Coefficiente_Tavola"], errors="coerce")
    return coeff_long.dropna(subset=["Coefficiente_Tavola", "Mese_Num"])


def _run_tavole(df: pd.DataFrame, ref_ym: int) -> pd.DataFrame:
    """
    Metodo TavoleStream originale:
      prezzo_att = importotrasp * (0.3 * carb_ref/carb_i + 0.7 * coeff_i/coeff_ref)

    Se Coefficiente (da TavoleStream) o prezzo_carb non sono nel dataset,
    li carica/ricostruisce da TavoleStream.xlsx e dal dataset stesso.
    """
    print(f"[TAVOLE] Caricamento TavoleStream da {TAVOLE_PATH}...")
    tavole = _load_tavole_coefficienti()
    print(f"  Righe coefficienti: {len(tavole)}")

    df = df.copy()

    # Merge Coefficiente_Tavola se necessario
    if "Coefficiente" not in df.columns or df["Coefficiente"].isna().all():
        print("  [TAVOLE] Merge coefficienti da TavoleStream.xlsx...")
        df = df.merge(
            tavole,
            left_on=["anno", "mese"],
            right_on=["Anno_Tavola", "Mese_Num"],
            how="left",
        )
        df = df.drop(columns=["Anno_Tavola", "Mese_Num"], errors="ignore")
        df["Coefficiente"] = df["Coefficiente_Tavola"]
        df = df.drop(columns=["Coefficiente_Tavola"], errors="ignore")
    else:
        # Se Coefficiente esiste gia' nel dataset, usa quello
        df = df.drop(columns=["Coefficiente_Tavola"], errors="ignore")

    before = len(df)
    df = df.dropna(subset=["Coefficiente"])
    if len(df) < before:
        print(f"  [TAVOLE] Righe senza coefficiente droppate: {before - len(df)}")

    # Verifica prezzo_carb
    if "prezzo_carb" not in df.columns or df["prezzo_carb"].isna().all():
        warnings.warn("[TAVOLE] Colonna prezzo_carb non disponibile. "
                       "Uso solo coefficiente (100% peso su Coefficiente).")
        # Fallback: solo coefficiente
        ref_row = df.sort_values(DATE_COL).iloc[-1]
        coeff_ref = float(ref_row["Coefficiente"])
        df["prezzo_attualizzato"] = (df[PRICE_COL] * (df["Coefficiente"] / coeff_ref)).round(0)
    else:
        # Formula originale 70/30
        df_sorted = df.sort_values(DATE_COL)
        carb_ref = float(df_sorted["prezzo_carb"].iloc[-1])
        coeff_ref = float(df_sorted["Coefficiente"].iloc[-1])
        print(f"  Prezzo carburante riferimento: {carb_ref}")
        print(f"  Coefficiente riferimento: {coeff_ref}")

        df["prezzo_attualizzato"] = (
            df[PRICE_COL] * (
                PESO_CARBURANTE * (carb_ref / df["prezzo_carb"])
                + PESO_COEFF * (df["Coefficiente"] / coeff_ref)
            )
        ).round(0)

    return df


# ╔════════════════════════════════════════════════════════════════╗
# ║  4. METODO "ipc_blend" — inflazione stimata + carburante     ║
# ╚════════════════════════════════════════════════════════════════╝
#
#  Idea:  il metodo chain calcola i rapporti mese→mese da trasporti
#         "simili" ma e' soggetto a rumore (pochi campioni per coppia).
#         Il metodo tavole dipende da un file esterno e dalla formula
#         fissa 70/30.
#
#  "ipc_blend" stima l'inflazione dei prezzi di trasporto direttamente
#  dal dataset, usando la mediana robusta per tipo_carico per ogni mese,
#  e la combina con il carburante (se disponibile).
#
#  Passi:
#    1. Per ogni tipo_carico, calcola la mediana mensile di importo_norm
#       (= prezzo normalizzato per km e spazio, gia' disponibile).
#    2. Smoothing con media mobile 3 mesi per ridurre il rumore.
#    3. Costruisce indice mensile (base=100 al mese di riferimento).
#    4. Se prezzo_carb disponibile: blend con indice carburante.
#    5. prezzo_att = importotrasp * (indice_ref / indice_mese)
#
#  Vantaggi rispetto a "chain":
#    - Usa TUTTI i dati, non solo i gruppi matched.
#    - Mediana robusta agli outlier (il chain usa la media).
#    - Smoothing riduce varianza mese-mese.
#

def _build_price_index(df: pd.DataFrame, ref_ym: int) -> pd.DataFrame:
    """
    Costruisce indice prezzi mensile per tipo_carico dalla mediana
    di importo_norm (normalizzato per km*spazio).
    Se importo_norm non e' disponibile, usa importo_per_km.
    """
    # Sceglie la metrica normalizzata migliore
    if "importo_norm" in df.columns and df["importo_norm"].notna().sum() > 100:
        norm_col = "importo_norm"
    elif "importo_per_km" in df.columns:
        norm_col = "importo_per_km"
    else:
        # Fallback: normalizza al volo
        df = df.copy()
        df["_norm_fallback"] = df[PRICE_COL] / df["km_tratta"].clip(lower=1)
        norm_col = "_norm_fallback"

    records = []
    for tipo in df["tipo_carico"].unique():
        sub = df[df["tipo_carico"] == tipo].copy()
        monthly_med = sub.groupby("ym")[norm_col].median()

        # Smoothing: media mobile centrata 3 mesi
        smoothed = monthly_med.rolling(3, center=True, min_periods=1).mean()

        for ym, val in smoothed.items():
            records.append({"tipo_carico": tipo, "ym": ym, "indice_trasporto": val})

    idx = pd.DataFrame(records)

    # Normalizza: base=100 al mese di riferimento
    for tipo in idx["tipo_carico"].unique():
        mask = (idx["tipo_carico"] == tipo)
        ref_mask = mask & (idx["ym"] == ref_ym)
        if ref_mask.any():
            base_val = idx.loc[ref_mask, "indice_trasporto"].values[0]
        else:
            # Se il mese ref non ha dati, usa l'ultimo mese disponibile
            sub_idx = idx.loc[mask].sort_values("ym")
            base_val = sub_idx["indice_trasporto"].iloc[-1]

        if base_val > 0:
            idx.loc[mask, "indice_trasporto"] = (
                idx.loc[mask, "indice_trasporto"] / base_val * 100.0
            )
        else:
            idx.loc[mask, "indice_trasporto"] = 100.0

    return idx


def _build_fuel_index(df: pd.DataFrame, ref_ym: int) -> pd.DataFrame:
    """Indice carburante mensile (base=100 al mese di riferimento)."""
    if "prezzo_carb" not in df.columns or df["prezzo_carb"].isna().all():
        return None

    fuel_monthly = df.groupby("ym")["prezzo_carb"].median()

    # Base 100 al mese di riferimento
    if ref_ym in fuel_monthly.index:
        base_fuel = fuel_monthly[ref_ym]
    else:
        base_fuel = fuel_monthly.iloc[-1]

    fuel_idx = (fuel_monthly / base_fuel * 100.0).reset_index()
    fuel_idx.columns = ["ym", "indice_carburante"]
    return fuel_idx


def _run_ipc_blend(df: pd.DataFrame, ref_ym: int) -> pd.DataFrame:
    """
    Attualizzazione IPC-blend:
      fattore = w_infl * (100/indice_trasporto) + w_fuel * (100/indice_carburante)
      prezzo_att = importotrasp * fattore
    """
    print("[IPC_BLEND] Costruzione indice prezzi trasporto...")
    price_idx = _build_price_index(df, ref_ym)
    print(f"  Righe indice: {len(price_idx)}")

    fuel_idx = _build_fuel_index(df, ref_ym)
    has_fuel = fuel_idx is not None
    if has_fuel:
        print(f"  Indice carburante disponibile ({len(fuel_idx)} mesi)")
        w_infl = IPC_PESO_INFLAZIONE
        w_fuel = IPC_PESO_CARBURANTE
    else:
        print("  Indice carburante NON disponibile — uso solo indice trasporto")
        w_infl = 1.0
        w_fuel = 0.0

    df = df.copy()

    # Merge indice trasporto (per tipo_carico + ym)
    df = df.merge(
        price_idx[["tipo_carico", "ym", "indice_trasporto"]],
        on=["tipo_carico", "ym"],
        how="left",
    )
    df["indice_trasporto"] = df["indice_trasporto"].fillna(100.0)

    # Merge indice carburante (per ym, globale)
    if has_fuel:
        df = df.merge(fuel_idx, on="ym", how="left")
        df["indice_carburante"] = df["indice_carburante"].fillna(100.0)
    else:
        df["indice_carburante"] = 100.0

    # Calcola fattore di attualizzazione
    df["fattore_att"] = (
        w_infl * (100.0 / df["indice_trasporto"].clip(lower=1.0))
        + w_fuel * (100.0 / df["indice_carburante"].clip(lower=1.0))
    )

    df["prezzo_attualizzato"] = (df[PRICE_COL] * df["fattore_att"]).round(0)

    # Log statistiche
    print(f"\n  Statistiche fattore attualizzazione:")
    print(f"    Media:   {df['fattore_att'].mean():.4f}")
    print(f"    Mediana: {df['fattore_att'].median():.4f}")
    print(f"    Min:     {df['fattore_att'].min():.4f}")
    print(f"    Max:     {df['fattore_att'].max():.4f}")

    # Export coefficienti (come tabella riassuntiva)
    coeff_summary = (
        df.groupby("ym")
        .agg(
            indice_trasporto_med=("indice_trasporto", "median"),
            indice_carburante_med=("indice_carburante", "median"),
            fattore_att_med=("fattore_att", "median"),
            n_righe=("fattore_att", "count"),
        )
        .reset_index()
    )
    coeff_summary["Anno"] = coeff_summary["ym"] // 100
    coeff_summary["Mese"] = coeff_summary["ym"] % 100
    coeff_path = "coefficienti_attualizzazione.xlsx"
    coeff_summary.to_excel(coeff_path, index=False)
    print(f"  Salvato: {coeff_path} ({len(coeff_summary)} righe)")

    # Cleanup colonne temporanee
    df = df.drop(columns=["indice_trasporto", "indice_carburante", "fattore_att"], errors="ignore")

    return df


# ╔════════════════════════════════════════════════════════════════╗
# ║  5. POST-PROCESSING COMUNE                                    ║
# ╚════════════════════════════════════════════════════════════════╝

def _recalculate_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Ricalcola metriche derivate con il prezzo attualizzato."""
    df = df.copy()

    if all(c in df.columns for c in ["prezzo_attualizzato", "km_tratta"]):
        df["importo_per_km"] = df["prezzo_attualizzato"] / df["km_tratta"]

    if all(c in df.columns for c in ["prezzo_attualizzato", "peso_totale"]):
        df["importo_per_peso"] = df["prezzo_attualizzato"] / df["peso_totale"]

    if all(c in df.columns for c in ["prezzo_attualizzato", "km_tratta", "spazio_calcolato"]):
        df["importo_norm"] = np.where(
            df["spazio_calcolato"] >= 1,
            1e5 * df["prezzo_attualizzato"] / (df["km_tratta"] * df["spazio_calcolato"]),
            np.nan,
        )

    return df


def _cleanup_temp_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rimuove colonne temporanee usate per l'attualizzazione."""
    temp_cols = ["anno", "mese", "ym", "coeff_att",
                 "_norm_fallback"]
    return df.drop(columns=[c for c in temp_cols if c in df.columns], errors="ignore")


def _export_coefficients(cumulative: dict, ref_ym: int,
                          path: str = "coefficienti_attualizzazione.xlsx"):
    """Export tabella coefficienti (usata dal metodo chain)."""
    result = pd.DataFrame([
        {"Anno": ym // 100, "Mese": ym % 100, "Coefficiente": round(cumulative[ym], 6)}
        for ym in sorted(cumulative.keys())
        if ym <= ref_ym
    ])
    result.to_excel(path, index=False)
    print(f"  Salvato: {path} ({len(result)} righe)")
    return result


# ╔════════════════════════════════════════════════════════════════╗
# ║  MAIN                                                         ║
# ╚════════════════════════════════════════════════════════════════╝

def main():
    input_file  = "02_preprocessed.xlsx"
    output_file = "03_attualizzato.xlsx"
    ref_ym = REF_YEAR * 100 + REF_MONTH

    print(f"[LOAD] Caricamento da {input_file}...")
    df = load_preprocessed(input_file)
    print(f"  Righe: {len(df)}")
    print(f"  Metodo di attualizzazione: {METODO}")
    print(f"  Mese di riferimento: {REF_YEAR}-{REF_MONTH:02d}")

    # ── Dispatch al metodo scelto ──
    if METODO == "chain":
        df_att = _run_chain(df, ref_ym)

    elif METODO == "tavole":
        df_att = _run_tavole(df, ref_ym)

    elif METODO == "ipc_blend":
        df_att = _run_ipc_blend(df, ref_ym)

    else:
        raise ValueError(f"Metodo sconosciuto: '{METODO}'. Usa 'chain', 'tavole' o 'ipc_blend'.")

    # ── Post-processing comune ──
    print("\n[POST] Ricalcolo metriche derivate...")
    df_att = _recalculate_derived_metrics(df_att)
    df_att = _cleanup_temp_columns(df_att)

    print(f"  Righe con prezzo_attualizzato: {df_att['prezzo_attualizzato'].notna().sum()}")

    # Statistiche per tipo_carico
    if "tipo_carico" in df_att.columns:
        print("\n  Mediana prezzo_attualizzato per tipo_carico:")
        for tipo, med in df_att.groupby("tipo_carico")["prezzo_attualizzato"].median().items():
            print(f"    {tipo}: {med:.0f}")

    # Confronto con prezzo originale
    if PRICE_COL in df_att.columns:
        ratio = df_att["prezzo_attualizzato"] / df_att[PRICE_COL].clip(lower=1)
        print(f"\n  Rapporto attualizzato/originale:")
        print(f"    Media:   {ratio.mean():.4f}")
        print(f"    Mediana: {ratio.median():.4f}")
        print(f"    Std:     {ratio.std():.4f}")

    # Export
    print("\n[EXPORT] Salvataggio...")
    with pd.ExcelWriter(output_file, engine="openpyxl", mode="w") as writer:
        df_att.to_excel(writer, sheet_name="Attualizzato", index=False)
    print(f"  Salvato: {output_file}")
    print(f"[OK] Attualizzazione completata (metodo: {METODO}).")


if __name__ == "__main__":
    main()
