import hmac
import json
import os
import re
from datetime import datetime
from pathlib import Path

import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

st.set_page_config(page_title="Amazon Warehouse Stock", layout="wide")

CITY_NAMES = {
    "DEL": "Delhi",
    "BOM": "Mumbai",
    "BLR": "Bangalore",
    "MAA": "Chennai",
    "HYD": "Hyderabad",
    "CCU": "Kolkata",
    "AMD": "Ahmedabad",
    "PNQ": "Pune",
    "JAI": "Jaipur",
    "LKO": "Lucknow",
    "CJB": "Coimbatore",
}

DATA_DIR = Path(__file__).parent / "data"
LOCATIONS_FILE = DATA_DIR / "warehouse_locations.json"
SETTINGS_FILE = DATA_DIR / "settings.json"
HISTORY_FILE = DATA_DIR / "history.json"

DEFAULT_SETTINGS = {
    "enabled": True,
    "global_threshold": 10,
    "scope": "per_city",
    "sku_overrides": {},
}


@st.cache_data
def load_warehouse_locations() -> dict[str, dict]:
    with open(LOCATIONS_FILE, encoding="utf-8") as f:
        return json.load(f)


def load_settings() -> dict:
    if SETTINGS_FILE.exists():
        with open(SETTINGS_FILE, encoding="utf-8") as f:
            saved = json.load(f)
        return {**DEFAULT_SETTINGS, **saved}
    return DEFAULT_SETTINGS.copy()


def save_settings(settings: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)


def load_history() -> list[dict]:
    if not HISTORY_FILE.exists():
        return []
    with open(HISTORY_FILE, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def save_history(history: list[dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def make_snapshot(agg: pd.DataFrame, ledger_data_date: str | None = None) -> dict:
    now = datetime.now()
    records = []
    has_transit = "In Transit Between Warehouses" in agg.columns
    for _, row in agg.iterrows():
        rec = {
            "Location": row["Location"],
            "CityCode": row["CityCode"],
            "MSKU": str(row["MSKU"]),
            "sellable": int(row["Ending Warehouse Balance"]),
        }
        if has_transit:
            rec["in_transit"] = int(row["In Transit Between Warehouses"])
        records.append(rec)

    return {
        "id": now.strftime("%Y%m%d_%H%M%S"),
        "uploaded_at": now.strftime("%Y-%m-%d %H:%M:%S"),
        "ledger_data_date": ledger_data_date,
        "records": records,
        "summary": {
            "warehouses": int(agg["Location"].nunique()),
            "mskus": int(agg["MSKU"].nunique()),
            "total_sellable": int(agg["Ending Warehouse Balance"].sum()),
            "total_in_transit": int(agg["In Transit Between Warehouses"].sum()) if has_transit else 0,
        },
    }


def append_snapshot(agg: pd.DataFrame, ledger_data_date: str | None = None) -> dict:
    snapshot = make_snapshot(agg, ledger_data_date)
    history = load_history()
    history.insert(0, snapshot)
    save_history(history)
    return snapshot


def delete_snapshot(snapshot_id: str) -> None:
    history = [s for s in load_history() if s["id"] != snapshot_id]
    save_history(history)


def get_snapshot(snapshot_id: str) -> dict | None:
    for snap in load_history():
        if snap["id"] == snapshot_id:
            return snap
    return None


def agg_from_snapshot(snapshot: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.DataFrame(snapshot["records"])
    rename = {"sellable": "Ending Warehouse Balance"}
    if "in_transit" in df.columns:
        rename["in_transit"] = "In Transit Between Warehouses"
    agg = df.rename(columns=rename)
    city_parts = {"Ending Warehouse Balance": "sum"}
    if "In Transit Between Warehouses" in agg.columns:
        city_parts["In Transit Between Warehouses"] = "sum"
    city_agg = agg.groupby(["CityCode", "MSKU"], as_index=False).agg(city_parts)
    return agg, city_agg


def parse_ledger_csv(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    df.columns = [c.strip() for c in df.columns]

    required = {
        "msku": "MSKU",
        "disposition": "Disposition",
        "balance": "Ending Warehouse Balance",
        "location": "Location",
    }
    found = {
        key: (_find_col(df, pretty) or _find_col(df, key))
        for key, pretty in required.items()
    }

    missing = [pretty for key, pretty in required.items() if found[key] is None]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}\nFound: {list(df.columns)}")

    msku_col = found["msku"]
    disp_col = found["disposition"]
    bal_col = found["balance"]
    loc_col = found["location"]
    transit_col = _find_col(df, "In Transit Between Warehouses")
    ship_col = _find_col(df, "Customer Shipments")

    df[bal_col] = (
        pd.to_numeric(
            df[bal_col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace(r"[^\d.\-]", "", regex=True),
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
    )
    df[disp_col] = df[disp_col].astype(str).str.strip().str.upper()
    df[loc_col] = df[loc_col].fillna("Unknown")

    df_sellable = df[df[disp_col].isin({"SELLABLE"}) & (df[bal_col] > 0)].copy()
    if df_sellable.empty:
        raise ValueError("No SELLABLE items with a positive balance found.")

    group_cols = [loc_col, msku_col]
    agg_parts = {bal_col: "sum"}
    if transit_col:
        agg_parts[transit_col] = "sum"

    agg = df_sellable.groupby(group_cols, as_index=False).agg(agg_parts)
    rename_map = {loc_col: "Location", msku_col: "MSKU", bal_col: "Ending Warehouse Balance"}
    if transit_col:
        rename_map[transit_col] = "In Transit Between Warehouses"
    agg.rename(columns=rename_map, inplace=True)

    velocity = None
    if ship_col:
        vel = (
            df_sellable.groupby(msku_col, as_index=False)[ship_col]
            .sum()
            .rename(columns={msku_col: "MSKU", ship_col: "Units Sold"})
        )
        vel["Units Sold"] = vel["Units Sold"].abs()
        velocity = vel

    agg["CityCode"] = agg["Location"].apply(extract_city_code)
    city_agg_parts = {"Ending Warehouse Balance": "sum"}
    if "In Transit Between Warehouses" in agg.columns:
        city_agg_parts["In Transit Between Warehouses"] = "sum"
    city_agg = agg.groupby(["CityCode", "MSKU"], as_index=False).agg(city_agg_parts)

    return agg, city_agg, velocity


def _read_csv_safe(uploaded_file):
    uploaded_file.seek(0)
    for enc in (None, "latin1", "utf-8"):
        try:
            if enc:
                return pd.read_csv(uploaded_file, encoding=enc)
            return pd.read_csv(uploaded_file)
        except Exception:
            uploaded_file.seek(0)
    uploaded_file.seek(0)
    return pd.read_csv(uploaded_file, engine="python", encoding="utf-8", on_bad_lines="skip")


def _find_col(df, name):
    mapping = {c.strip().lower(): c for c in df.columns}
    return mapping.get(name.strip().lower())


def format_data_date(value) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        parsed = pd.to_datetime(text, dayfirst=True, errors="coerce")
    if pd.isna(parsed):
        return None
    return f"{int(parsed.day)} {parsed.strftime('%B').upper()} {parsed.year}"


def extract_ledger_data_date(df: pd.DataFrame) -> str | None:
    date_col = _find_col(df, "Date")
    if not date_col:
        return None
    dates = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    if dates.isna().all():
        return None
    return format_data_date(dates.max())


def snapshot_recorded_date(snapshot: dict) -> str | None:
    return format_data_date(snapshot.get("uploaded_at"))


def snapshot_ledger_data_date(snapshot: dict) -> str | None:
    return snapshot.get("ledger_data_date")


def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def extract_city_code(location: str) -> str:
    return re.sub(r"\d+$", "", str(location)).strip().upper()


def city_display_name(code: str) -> str:
    return CITY_NAMES.get(code, code)


def threshold_for_msku(msku: str, settings: dict) -> int:
    overrides = settings.get("sku_overrides") or {}
    return int(overrides.get(str(msku).strip(), settings.get("global_threshold", 10)))


def compute_low_stock_alerts(agg: pd.DataFrame, settings: dict) -> dict:
    empty = {
        "flagged_skus": set(),
        "flagged_locations": set(),
        "flagged_cities": set(),
        "alerts": [],
        "location_alert_count": {},
        "city_alert_count": {},
    }
    if not settings.get("enabled", True):
        return empty

    expanded = expand_city_warehouse_stock(agg)
    flagged_skus: set[tuple[str, str]] = set()
    alerts: list[dict] = []

    for _, row in expanded.iterrows():
        loc = row["Location"]
        msku = str(row["MSKU"])
        qty = int(row["Ending Warehouse Balance"])
        thresh = threshold_for_msku(msku, settings)
        if qty <= thresh:
            flagged_skus.add((loc, msku))
            alerts.append(
                {
                    "MSKU": msku,
                    "scope": "Per warehouse",
                    "location": loc,
                    "qty": qty,
                    "threshold": thresh,
                    "status": "LOW",
                }
            )

    flagged_locations: set[str] = set()
    flagged_cities: set[str] = set()
    location_alert_count: dict[str, int] = {}
    city_alert_count: dict[str, int] = {}

    for loc, msku in flagged_skus:
        flagged_locations.add(loc)
        location_alert_count[loc] = location_alert_count.get(loc, 0) + 1
        city = extract_city_code(loc)
        flagged_cities.add(city)
        city_alert_count[city] = city_alert_count.get(city, 0) + 1

    alerts.sort(key=lambda a: (a["location"], a["MSKU"]))

    return {
        "flagged_skus": flagged_skus,
        "flagged_locations": flagged_locations,
        "flagged_cities": flagged_cities,
        "alerts": alerts,
        "location_alert_count": location_alert_count,
        "city_alert_count": city_alert_count,
    }


def is_sku_flagged(
    loc: str, city_code: str, msku: str, scope: str, alerts_data: dict
) -> bool:
    return (loc, str(msku)) in alerts_data["flagged_skus"]


def is_sku_flagged_in_city(
    city_code: str, msku: str, scope: str, alerts_data: dict, agg: pd.DataFrame
) -> bool:
    locs = agg.loc[agg["CityCode"] == city_code, "Location"].unique()
    msku = str(msku)
    return any((loc, msku) in alerts_data["flagged_skus"] for loc in locs)


def build_sku_threshold_table(agg: pd.DataFrame, settings: dict) -> pd.DataFrame:
    totals = (
        agg.groupby("MSKU", as_index=False)["Ending Warehouse Balance"]
        .sum()
        .rename(columns={"Ending Warehouse Balance": "Total Stock"})
    )
    totals["MSKU"] = totals["MSKU"].astype(str)
    totals["Threshold"] = totals["MSKU"].apply(lambda m: threshold_for_msku(m, settings))
    overrides = settings.get("sku_overrides") or {}
    totals["Override"] = totals["MSKU"].map(lambda m: overrides.get(m, ""))
    totals["Override"] = totals["Override"].apply(lambda v: int(v) if v != "" else "")
    totals["Status"] = totals.apply(
        lambda r: "LOW (national)" if r["Total Stock"] <= r["Threshold"] else "OK", axis=1
    )
    return totals.sort_values("Total Stock")


def build_scope_status_table(agg: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """Per-SKU status at the active scope (city or warehouse)."""
    scope = settings.get("scope", "per_city")
    rows: list[dict] = []

    for msku in sorted(agg["MSKU"].astype(str).unique()):
        thresh = threshold_for_msku(msku, settings)
        msku_rows = agg[agg["MSKU"].astype(str) == msku]

        if scope == "per_warehouse":
            for loc, grp in msku_rows.groupby("Location"):
                qty = int(grp["Ending Warehouse Balance"].sum())
                rows.append(
                    {
                        "MSKU": msku,
                        "Location": loc,
                        "Qty": qty,
                        "Threshold": thresh,
                        "Status": "LOW" if qty <= thresh else "OK",
                    }
                )
        else:
            for city, grp in msku_rows.groupby("CityCode"):
                qty = int(grp["Ending Warehouse Balance"].sum())
                rows.append(
                    {
                        "MSKU": msku,
                        "Location": f"{city_display_name(city)} ({city})",
                        "Qty": qty,
                        "Threshold": thresh,
                        "Status": "LOW" if qty <= thresh else "OK",
                    }
                )

    return pd.DataFrame(rows).sort_values(["Status", "Qty"])


def expand_city_warehouse_stock(agg: pd.DataFrame) -> pd.DataFrame:
    """All warehouses in the upload × all MSKUs present anywhere, zero-filled where absent."""
    stock = (
        agg.groupby(["Location", "CityCode", "MSKU"], as_index=False)["Ending Warehouse Balance"]
        .sum()
    )
    stock["MSKU"] = stock["MSKU"].astype(str)
    warehouses = stock[["Location", "CityCode"]].drop_duplicates().sort_values(["CityCode", "Location"])
    all_mskus = sorted(stock["MSKU"].unique())
    rows: list[dict] = []

    for _, wh in warehouses.iterrows():
        loc = wh["Location"]
        city = wh["CityCode"]
        for msku in all_mskus:
            match = stock[(stock["Location"] == loc) & (stock["MSKU"] == msku)]
            qty = int(match["Ending Warehouse Balance"].iloc[0]) if not match.empty else 0
            rows.append(
                {
                    "Location": loc,
                    "CityCode": city,
                    "MSKU": msku,
                    "Ending Warehouse Balance": qty,
                }
            )

    return pd.DataFrame(rows)


def build_send_plan(agg: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """Warehouse-level send plan — zero stock shown at every FC missing an MSKU."""
    expanded = expand_city_warehouse_stock(agg)
    rows: list[dict] = []

    for _, row in expanded.iterrows():
        loc = row["Location"]
        city = row["CityCode"]
        msku = str(row["MSKU"])
        current = int(row["Ending Warehouse Balance"])
        threshold = threshold_for_msku(msku, settings)
        shortfall = max(0, threshold - current)
        shortfall_pct = (shortfall / threshold * 100) if threshold > 0 else (100.0 if shortfall else 0.0)
        rows.append(
            {
                "City": f"{city_display_name(city)} ({city})",
                "CityCode": city,
                "Warehouse": loc,
                "MSKU": msku,
                "Current": current,
                "Threshold": threshold,
                "Shortfall": shortfall,
                "Shortfall %": round(shortfall_pct, 1),
                "Send qty": 0,
                "_low": current <= threshold,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(["Shortfall %", "Shortfall"], ascending=[False, False]).reset_index(drop=True)
    df.insert(0, "Priority", range(1, len(df) + 1))
    return df


def aggregate_send_plan_by_city(plan: pd.DataFrame) -> pd.DataFrame:
    """Roll warehouse rows up to city + MSKU for city-level send planning."""
    if plan.empty:
        return plan.copy()

    grouped = (
        plan.groupby(["City", "CityCode", "MSKU"], as_index=False)
        .agg(
            Current=("Current", "sum"),
            Threshold=("Threshold", "first"),
            Send_qty=("Send qty", "sum"),
        )
        .rename(columns={"Send_qty": "Send qty"})
    )
    grouped["Shortfall"] = (grouped["Threshold"] - grouped["Current"]).clip(lower=0).astype(int)
    grouped["Shortfall %"] = grouped.apply(
        lambda r: round(r["Shortfall"] / r["Threshold"] * 100, 1)
        if r["Threshold"] > 0
        else (100.0 if r["Shortfall"] else 0.0),
        axis=1,
    )
    grouped["_low"] = grouped["Current"] <= grouped["Threshold"]
    grouped = grouped.sort_values(["Shortfall %", "Shortfall"], ascending=[False, False]).reset_index(drop=True)
    grouped.insert(0, "Priority", range(1, len(grouped) + 1))
    return grouped


def send_plan_summary(plan: pd.DataFrame) -> dict:
    if plan.empty:
        return {"cities": 0, "skus": 0, "units_short": 0, "warehouses": 0}
    low = plan[plan["_low"]] if "_low" in plan.columns else plan[plan["Shortfall"] > 0]
    return {
        "cities": int(low["CityCode"].nunique()) if not low.empty else 0,
        "warehouses": int(low["Warehouse"].nunique()) if not low.empty else 0,
        "skus": len(low),
        "units_short": int(low["Shortfall"].sum()) if not low.empty else 0,
    }


def _sku_table_html(
    agg_slice: pd.DataFrame,
    alerts_data: dict,
    scope_key: str,
    is_city: bool,
    scope: str,
) -> str:
    rows = agg_slice.sort_values("Ending Warehouse Balance", ascending=False)
    lines = [
        "<table style='font-size:12px;border-collapse:collapse;width:100%'>",
        "<tr><th style='text-align:left;padding:2px 6px'>MSKU</th>"
        "<th style='text-align:right;padding:2px 6px'>Sellable</th>"
        "<th style='padding:2px 6px'>Alert</th></tr>",
    ]
    for _, r in rows.iterrows():
        msku = str(r["MSKU"])
        qty = int(r["Ending Warehouse Balance"])
        if is_city:
            if scope == "per_city":
                flagged = (scope_key, msku) in alerts_data["flagged_skus"]
            else:
                flagged = any(
                    extract_city_code(loc) == scope_key
                    for loc, m in alerts_data["flagged_skus"]
                    if m == msku
                )
        else:
            city = extract_city_code(scope_key)
            flagged = is_sku_flagged(scope_key, city, msku, scope, alerts_data)
        mark = "⚠️" if flagged else ""
        lines.append(
            f"<tr><td style='padding:2px 6px'>{msku}</td>"
            f"<td style='text-align:right;padding:2px 6px'>{qty:,}</td>"
            f"<td style='padding:2px 6px'>{mark}</td></tr>"
        )
    lines.append("</table>")
    return "".join(lines)


def generate_html_report(agg_df, city_agg, timestamp):
    def tbl(headers, rows, hcolor="#2c6fad"):
        cols = "".join(f"<th>{h}</th>" for h in headers)
        body = "".join(
            "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows
        )
        return (
            f'<table><thead style="background:{hcolor};color:white"><tr>{cols}</tr></thead>'
            f"<tbody>{body}</tbody></table>"
        )

    css = """<style>
body{font-family:Arial,sans-serif;font-size:13px;color:#222;margin:24px}
h1{color:#1a3e6e}h2{color:#2c6fad;border-bottom:2px solid #2c6fad;padding-bottom:4px;margin-top:32px}
h3{color:#444;margin-bottom:4px}
table{border-collapse:collapse;width:100%;margin-bottom:20px}
th,td{border:1px solid #ccc;padding:6px 10px;text-align:left}
tr:nth-child(even){background:#f5f8fc}
</style>"""

    html = (
        f'<!DOCTYPE html><html><head><meta charset="utf-8"><title>Warehouse Stock Report</title>{css}</head><body>'
        f"<h1>Amazon Warehouse Stock Report</h1><p>Generated: <strong>{timestamp}</strong></p>"
    )

    html += "<h2>Units Per SKU Per City</h2>"
    for city_code, grp in city_agg.groupby("CityCode"):
        name = city_display_name(city_code)
        total = int(grp["Ending Warehouse Balance"].sum())
        html += f"<h3>{name} ({city_code}) — {total} units</h3>"
        rows = [
            [r["MSKU"], int(r["Ending Warehouse Balance"])]
            for _, r in grp.sort_values("Ending Warehouse Balance", ascending=False).iterrows()
        ]
        html += tbl(["MSKU", "Units"], rows)

    html += "<h2>Stock By Warehouse</h2>"
    for location, grp in agg_df.groupby("Location"):
        total = int(grp["Ending Warehouse Balance"].sum())
        transit = (
            int(grp["In Transit Between Warehouses"].sum())
            if "In Transit Between Warehouses" in grp.columns
            else 0
        )
        html += f"<h3>{location} — {total} sellable · {transit} in transit</h3>"
        rows = [
            [
                r["MSKU"],
                int(r.get("In Transit Between Warehouses", 0)),
                int(r["Ending Warehouse Balance"]),
            ]
            for _, r in grp.sort_values("Ending Warehouse Balance", ascending=False).iterrows()
        ]
        html += tbl(["MSKU", "In Transit", "Sellable Units"], rows, "#555")

    html += "</body></html>"
    return html.encode("utf-8")


def _city_has_alerts(city_code: str, alerts_data: dict, agg: pd.DataFrame) -> bool:
    if city_code in alerts_data["flagged_cities"]:
        return True
    if alerts_data["city_alert_count"].get(city_code, 0) > 0:
        return True
    city_locs = agg.loc[agg["CityCode"] == city_code, "Location"].unique()
    return any(
        loc in alerts_data["flagged_locations"]
        or alerts_data["location_alert_count"].get(loc, 0) > 0
        for loc in city_locs
    )


def _city_alert_total(city_code: str, alerts_data: dict, agg: pd.DataFrame) -> int:
    if city_code in alerts_data["city_alert_count"]:
        return alerts_data["city_alert_count"][city_code]
    total = 0
    for loc in agg.loc[agg["CityCode"] == city_code, "Location"].unique():
        total += alerts_data["location_alert_count"].get(loc, 0)
    return total


def _location_has_alerts(loc: str, city_code: str, alerts_data: dict) -> bool:
    if loc in alerts_data["flagged_locations"]:
        return True
    if alerts_data["location_alert_count"].get(loc, 0) > 0:
        return True
    if city_code in alerts_data["flagged_cities"]:
        return True
    return False


def _location_totals(agg: pd.DataFrame) -> pd.DataFrame:
    totals = (
        agg.groupby("Location", as_index=False)["Ending Warehouse Balance"]
        .sum()
        .rename(columns={"Ending Warehouse Balance": "sellable"})
    )
    totals["CityCode"] = totals["Location"].apply(extract_city_code)
    return totals


def build_warehouse_map(
    agg: pd.DataFrame,
    locations_db: dict,
    mode: str,
    selected_cities: list[str] | None,
    alerts_data: dict | None = None,
    show_low_stock_only: bool = False,
    scope: str = "per_city",
) -> tuple[folium.Map, list[str]]:
    alerts_data = alerts_data or compute_low_stock_alerts(agg, {"enabled": False})
    loc_totals = _location_totals(agg)
    unmapped = [loc for loc in loc_totals["Location"] if loc not in locations_db]

    m = folium.Map(location=[20.6, 78.9], zoom_start=4, tiles="OpenStreetMap")

    if mode == "All warehouses":
        for _, row in loc_totals.iterrows():
            loc = row["Location"]
            city_code = row["CityCode"]
            if selected_cities and city_code not in selected_cities:
                continue
            meta = locations_db.get(loc)
            if not meta:
                continue

            is_low = _location_has_alerts(loc, city_code, alerts_data)
            if show_low_stock_only and not is_low:
                continue

            sellable = int(row["sellable"])
            lat, lng = meta["lat"], meta["lng"]

            alert_n = alerts_data["location_alert_count"].get(loc, 0)
            sku_table = _sku_table_html(
                agg[agg["Location"] == loc].groupby("MSKU", as_index=False)[
                    "Ending Warehouse Balance"
                ].sum(),
                alerts_data,
                loc,
                is_city=False,
                scope=scope,
            )
            alert_line = f"<br><b>⚠️ Alerts:</b> {alert_n}" if alert_n else ""
            popup_html = (
                f"<b>{loc}</b><br>"
                f"{meta.get('address', '')}<br>"
                f"{meta.get('city', '')}, {meta.get('state', '')}<br>"
                f"<b>Sellable:</b> {sellable:,}{alert_line}<br><br>{sku_table}"
            )
            color = "#d9534f" if is_low else "#2c6fad"
            fill = "#e74c3c" if is_low else "#4a90d9"
            radius = 13 if is_low else 10
            tooltip = f"{loc} — {sellable:,} sellable"
            if alert_n:
                tooltip += f" · ⚠️ {alert_n} alert{'s' if alert_n != 1 else ''}"

            folium.CircleMarker(
                location=[lat, lng],
                radius=radius,
                color=color,
                fill=True,
                fill_color=fill,
                fill_opacity=0.85,
                tooltip=tooltip,
                popup=folium.Popup(popup_html, max_width=400),
            ).add_to(m)
    else:
        filtered = loc_totals
        if selected_cities:
            filtered = filtered[filtered["CityCode"].isin(selected_cities)]

        for city_code, grp in filtered.groupby("CityCode"):
            mapped = grp[grp["Location"].isin(locations_db)]
            if mapped.empty:
                continue

            is_low = _city_has_alerts(city_code, alerts_data, agg)
            if show_low_stock_only and not is_low:
                continue

            coords = [
                (locations_db[loc]["lat"], locations_db[loc]["lng"])
                for loc in mapped["Location"]
            ]
            lat = sum(c[0] for c in coords) / len(coords)
            lng = sum(c[1] for c in coords) / len(coords)

            total_sellable = int(mapped["sellable"].sum())
            fc_count = len(mapped)
            city_name = city_display_name(city_code)
            alert_n = _city_alert_total(city_code, alerts_data, agg)

            city_sku = (
                agg[agg["CityCode"] == city_code]
                .groupby("MSKU", as_index=False)["Ending Warehouse Balance"]
                .sum()
            )
            sku_table = _sku_table_html(city_sku, alerts_data, city_code, is_city=True, scope=scope)

            breakdown = " · ".join(
                f"{r['Location']}: {int(r['sellable']):,}"
                for _, r in mapped.sort_values("sellable", ascending=False).iterrows()
            )
            alert_line = f"<br><b>⚠️ Alerts:</b> {alert_n}" if alert_n else ""
            popup_html = (
                f"<b>{city_name}</b> ({city_code})<br>"
                f"<b>Sellable:</b> {total_sellable:,}{alert_line}<br>"
                f"<b>Warehouses ({fc_count}):</b> {breakdown}<br><br>{sku_table}"
            )
            color = "#d9534f" if is_low else "#1a3e6e"
            fill = "#e74c3c" if is_low else "#2c6fad"
            radius = 16 if is_low else 14
            tooltip = f"{city_name} — {total_sellable:,} sellable ({fc_count} warehouses)"
            if alert_n:
                tooltip += f" · ⚠️ {alert_n} alert{'s' if alert_n != 1 else ''}"

            folium.CircleMarker(
                location=[lat, lng],
                radius=radius,
                color=color,
                fill=True,
                fill_color=fill,
                fill_opacity=0.9,
                tooltip=tooltip,
                popup=folium.Popup(popup_html, max_width=420),
            ).add_to(m)

    return m, unmapped


def render_settings_panel(settings: dict, agg: pd.DataFrame | None = None) -> dict:
    st.subheader("⚙️ Alert thresholds")
    st.caption("Each warehouse is checked separately. If an MSKU appears anywhere, every FC in the upload is listed (0 where absent).")

    settings["enabled"] = st.checkbox(
        "Enable low-stock alerts",
        value=settings.get("enabled", True),
    )
    settings["global_threshold"] = st.number_input(
        "Default threshold (all MSKUs)",
        min_value=0,
        value=int(settings.get("global_threshold", 10)),
    )
    settings["scope"] = "per_city"

    if agg is not None:
        st.divider()
        st.subheader("📋 Per-MSKU overrides")
        sku_list = sorted(agg["MSKU"].astype(str).unique())
        overrides = dict(settings.get("sku_overrides") or {})

        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            pick_msku = st.selectbox("MSKU", options=sku_list, key="settings_pick_msku")
        with c2:
            pick_thresh = st.number_input(
                "Threshold",
                min_value=0,
                value=int(overrides.get(pick_msku, settings["global_threshold"])),
                key="settings_pick_thresh",
            )
        with c3:
            st.write("")
            st.write("")
            if st.button("Set override", key="settings_set_override"):
                overrides[pick_msku] = int(pick_thresh)
                settings["sku_overrides"] = overrides
                save_settings(settings)
                st.rerun()

        if overrides:
            for msku in sorted(overrides):
                rc1, rc2 = st.columns([4, 1])
                rc1.markdown(f"**{msku}** → {overrides[msku]} units")
                if rc2.button("Remove", key=f"settings_rm_{msku}"):
                    del overrides[msku]
                    settings["sku_overrides"] = overrides
                    save_settings(settings)
                    st.rerun()
            if st.button("Clear all overrides"):
                settings["sku_overrides"] = {}
                save_settings(settings)
                st.rerun()
        else:
            st.caption("No overrides — all MSKUs use the default threshold.")

    sc1, sc2 = st.columns(2)
    with sc1:
        if st.button("Save settings", type="primary"):
            st.session_state.settings = settings
            save_settings(settings)
            st.success("Settings saved.")
    with sc2:
        if st.button("Reset to defaults"):
            settings = DEFAULT_SETTINGS.copy()
            st.session_state.settings = settings
            save_settings(settings)
            st.rerun()

    st.session_state.settings = settings
    return settings


def render_history_panel() -> None:
    st.subheader("📜 Saved snapshots")
    history = load_history()
    if not history:
        st.info("No saved snapshots yet. Upload a CSV and choose **Save to history**.")
        return

    active_id = st.session_state.get("active_snapshot_id")
    for snap in history:
        summary = snap.get("summary", {})
        is_active = snap["id"] == active_id
        label = f"{'▶ ' if is_active else ''}{snap['uploaded_at']}"
        detail = (
            f"{summary.get('total_sellable', '?')} units · "
            f"{summary.get('warehouses', '?')} warehouses · "
            f"{summary.get('mskus', '?')} MSKUs"
        )
        hc1, hc2 = st.columns([5, 1])
        if hc1.button(label, key=f"tab_hist_{snap['id']}", help=detail, use_container_width=True):
            st.session_state.active_snapshot_id = snap["id"]
            st.session_state.prefer_history = True
            st.session_state.save_decision_for = None
            st.rerun()
        if hc2.button("🗑", key=f"tab_del_{snap['id']}"):
            delete_snapshot(snap["id"])
            if active_id == snap["id"]:
                st.session_state.active_snapshot_id = None
            st.rerun()


def render_summary_banner(plan: pd.DataFrame) -> None:
    summary = send_plan_summary(plan)
    if summary["skus"] == 0:
        st.success("All MSKUs are above threshold at city level — no replenishment needed.")
    else:
        st.warning(
            f"**{summary['cities']}** cities · "
            f"**{summary['warehouses']}** warehouses · "
            f"**{summary['skus']}** lines below threshold · "
            f"**{summary['units_short']}** units short"
        )


def prepare_send_plan_table(
    plan: pd.DataFrame,
    selected_skus: list[str],
    view_mode: str,
    show_all: bool,
    sort_by: str,
    sort_asc: bool,
) -> tuple[pd.DataFrame, list[str]]:
    view = plan.drop(columns=["_low"], errors="ignore")
    view = view[view["MSKU"].astype(str).isin(selected_skus)]

    if view_mode == "City":
        view = aggregate_send_plan_by_city(view)
        display_cols = ["Priority", "City", "MSKU", "Current", "Threshold", "Send qty"]
    else:
        display_cols = ["Priority", "City", "Warehouse", "MSKU", "Current", "Threshold", "Send qty"]

    if not show_all:
        view = view[view["Shortfall"] > 0]

    if view.empty:
        return view, display_cols

    if sort_by not in display_cols:
        sort_by = "Priority"
    view = view.sort_values(sort_by, ascending=sort_asc, kind="mergesort").reset_index(drop=True)
    view["Priority"] = range(1, len(view) + 1)
    return view[display_cols].copy(), display_cols


def render_send_plan_tab(plan: pd.DataFrame, source_key: str) -> None:
    st.subheader("📋 Send Plan")
    st.caption("Review shortages and enter how many units you plan to send. Export matches the table row order below.")

    all_skus = sorted(plan["MSKU"].astype(str).unique())
    fc1, fc2, fc3 = st.columns([2, 1, 1])
    with fc1:
        selected_skus = st.multiselect(
            "SKUs to include",
            options=all_skus,
            default=all_skus,
            help="Exclude SKUs you don't plan to replenish this round.",
            key=f"send_plan_skus_{source_key}",
        )
    with fc2:
        view_mode = st.radio(
            "View by",
            options=["City", "Warehouse"],
            index=0,
            horizontal=True,
            key=f"send_plan_view_{source_key}",
        )
    with fc3:
        show_all = st.checkbox("Show all SKUs (including above threshold)", value=False)

    if not selected_skus:
        st.warning("Select at least one MSKU to include in the send plan.")
        return

    display_cols = (
        ["Priority", "City", "MSKU", "Current", "Threshold", "Send qty"]
        if view_mode == "City"
        else ["Priority", "City", "Warehouse", "MSKU", "Current", "Threshold", "Send qty"]
    )
    sc1, sc2 = st.columns([2, 1])
    with sc1:
        sort_by = st.selectbox(
            "Sort by",
            options=display_cols,
            index=0,
            key=f"send_plan_sort_{view_mode.lower()}_{source_key}",
        )
    with sc2:
        sort_asc = st.toggle(
            "Ascending",
            value=True,
            key=f"send_plan_sort_asc_{view_mode.lower()}_{source_key}",
        )

    table, display_cols = prepare_send_plan_table(
        plan, selected_skus, view_mode, show_all, sort_by, sort_asc
    )
    if table.empty:
        st.info("No rows match your filters.")
        return

    data_key = f"send_plan_table_{view_mode.lower()}_{source_key}"
    meta_key = f"{data_key}_meta"
    editor_key = f"send_plan_editor_{view_mode.lower()}_{source_key}"
    table_meta = (view_mode, tuple(sorted(selected_skus)), show_all, sort_by, sort_asc)

    if st.session_state.get(meta_key) != table_meta:
        st.session_state[data_key] = table
        st.session_state[meta_key] = table_meta
        st.session_state.pop(editor_key, None)

    column_config = {
        "Priority": st.column_config.NumberColumn("Priority", disabled=True),
        "City": st.column_config.TextColumn("City", disabled=True),
        "MSKU": st.column_config.TextColumn("MSKU", disabled=True),
        "Current": st.column_config.NumberColumn("Current", disabled=True),
        "Threshold": st.column_config.NumberColumn("Threshold", disabled=True),
        "Send qty": st.column_config.NumberColumn("Send qty", min_value=0, step=1),
    }
    if view_mode == "Warehouse":
        column_config["Warehouse"] = st.column_config.TextColumn("Warehouse", disabled=True)

    edited = st.data_editor(
        st.session_state[data_key],
        column_config=column_config,
        hide_index=True,
        use_container_width=True,
        key=editor_key,
    )

    st.download_button(
        label="⬇️ Export send plan CSV",
        data=edited[display_cols].to_csv(index=False).encode("utf-8"),
        file_name=f"send_plan_{view_mode.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )


def render_overview_tab(
    agg: pd.DataFrame,
    city_agg: pd.DataFrame,
    velocity: pd.DataFrame | None,
    settings: dict,
    alerts_data: dict,
) -> None:
    scope = "per_city"
    location_totals = agg.groupby("Location")["Ending Warehouse Balance"].sum()
    total_in_transit = (
        int(agg["In Transit Between Warehouses"].sum())
        if "In Transit Between Warehouses" in agg.columns
        else 0
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Warehouses", agg["Location"].nunique())
    c2.metric("Unique MSKUs", agg["MSKU"].nunique())
    c3.metric("Total Sellable", int(agg["Ending Warehouse Balance"].sum()))
    c4.metric("In Transit", total_in_transit)

    if settings.get("enabled") and alerts_data["alerts"]:
        st.subheader(f"⚠️ Low Stock Alerts ({len(alerts_data['alerts'])})")
        st.dataframe(pd.DataFrame(alerts_data["alerts"]), use_container_width=True, hide_index=True)

    st.subheader("🏙️ Stock by city")
    city_order = city_agg.groupby("CityCode")["Ending Warehouse Balance"].sum().sort_values(ascending=False)
    flagged_cities = set(alerts_data["flagged_cities"])
    sorted_cities = list(flagged_cities) + [c for c in city_order.index if c not in flagged_cities]

    for city_chunk in chunks(sorted_cities, 3):
        cols = st.columns(3)
        for i, city_code in enumerate(city_chunk):
            with cols[i]:
                cdf = city_agg[city_agg["CityCode"] == city_code].sort_values(
                    "Ending Warehouse Balance", ascending=False
                )
                city_total = int(cdf["Ending Warehouse Balance"].sum())
                city_warn = " ⚠️" if city_code in flagged_cities else ""
                st.markdown(f"### {city_display_name(city_code)}{city_warn}")
                st.caption(f"**{city_total}** units sellable")
                for _, row in cdf.iterrows():
                    flagged = is_sku_flagged_in_city(city_code, row["MSKU"], scope, alerts_data, agg)
                    icon = "⚠️" if flagged else "📦"
                    st.markdown(f"{icon} **{row['MSKU']}** — {int(row['Ending Warehouse Balance'])}")

    if velocity is not None and not velocity.empty:
        with st.expander("📈 Units sold this period"):
            st.dataframe(velocity.sort_values("Units Sold", ascending=False), hide_index=True)

    with st.expander("🏬 Stock by warehouse"):
        for loc in location_totals.sort_values(ascending=False).index:
            loc_df = agg[agg["Location"] == loc].sort_values("Ending Warehouse Balance", ascending=False)
            loc_total = int(loc_df["Ending Warehouse Balance"].sum())
            loc_warn = " ⚠️" if loc in alerts_data["flagged_locations"] else ""
            st.markdown(f"**{loc}**{loc_warn} — {loc_total} units")
            for _, row in loc_df.iterrows():
                city_code = extract_city_code(loc)
                flagged = is_sku_flagged(loc, city_code, row["MSKU"], scope, alerts_data)
                icon = "⚠️" if flagged else "·"
                st.markdown(f"{icon} {row['MSKU']}: {int(row['Ending Warehouse Balance'])}")

    st.download_button(
        "⬇️ Download aggregated CSV",
        data=agg.to_csv(index=False).encode("utf-8"),
        file_name="aggregated_sellable_by_location_msku.csv",
        mime="text/csv",
    )


def render_map_tab(agg: pd.DataFrame, settings: dict, alerts_data: dict) -> None:
    st.subheader("🗺️ Warehouse map")
    locations_db = load_warehouse_locations()
    city_options = sorted(agg["CityCode"].unique())
    city_labels = {c: city_display_name(c) for c in city_options}

    mc1, mc2, mc3 = st.columns([1, 2, 1])
    with mc1:
        map_mode = st.radio("View", ["By city", "All warehouses"], horizontal=True, index=0)
    with mc2:
        selected_cities = st.multiselect(
            "Cities",
            city_options,
            default=city_options,
            format_func=lambda c: f"{city_labels[c]} ({c})",
        )
    with mc3:
        show_low_only = st.checkbox("Low stock only", value=True)

    warehouse_map, unmapped = build_warehouse_map(
        agg,
        locations_db,
        map_mode,
        selected_cities or None,
        alerts_data=alerts_data,
        show_low_stock_only=show_low_only,
        scope="per_city",
    )
    if unmapped:
        st.warning(f"Unmapped warehouses: {', '.join(unmapped)}")
    if selected_cities:
        st_folium(warehouse_map, width="100%", height=750, returned_objects=[])


def render_data_dates_banner(
    recorded_date: str | None,
    ledger_data_date: str | None,
) -> None:
    parts: list[str] = []
    if recorded_date:
        parts.append(f"Data recorded: <strong>{recorded_date}</strong>")
    if ledger_data_date:
        parts.append(f"DATA DATE: <strong>{ledger_data_date}</strong>")
    if not parts:
        return
    st.markdown(
        f"<div style='background:#1f2937;color:#fff;padding:14px 18px;border-radius:8px;"
        f"font-size:1.25rem;margin:0 0 16px 0;'>"
        f"📅 {' &nbsp;|&nbsp; '.join(parts)}</div>",
        unsafe_allow_html=True,
    )


def render_app(
    agg: pd.DataFrame,
    city_agg: pd.DataFrame,
    velocity: pd.DataFrame | None,
    settings: dict,
    source_label: str,
    source_key: str,
    recorded_date: str | None = None,
    ledger_data_date: str | None = None,
) -> None:
    render_data_dates_banner(recorded_date, ledger_data_date)

    st.caption(f"Data source: **{source_label}**")

    settings = st.session_state.settings
    settings["scope"] = "per_city"
    alerts_data = compute_low_stock_alerts(agg, settings)
    send_plan = build_send_plan(agg, settings)

    render_summary_banner(send_plan)

    tab_send, tab_overview, tab_map, tab_history = st.tabs(
        ["📋 Send Plan", "📊 Overview", "🗺️ Map", "⚙️ History & Settings"]
    )

    with tab_send:
        render_send_plan_tab(send_plan, source_key)

    with tab_overview:
        render_overview_tab(agg, city_agg, velocity, settings, alerts_data)

    with tab_map:
        render_map_tab(agg, settings, alerts_data)

    with tab_history:
        render_history_panel()
        st.divider()
        settings = render_settings_panel(settings, agg)


def get_app_password() -> str | None:
    try:
        if "APP_PASSWORD" in st.secrets:
            return str(st.secrets["APP_PASSWORD"])
    except (FileNotFoundError, KeyError):
        pass
    return os.environ.get("APP_PASSWORD")


def require_auth() -> None:
    if st.session_state.get("authenticated"):
        return

    password = get_app_password()
    if not password:
        st.title("🔒 Login")
        st.error(
            "App password is not configured. "
            "Set `APP_PASSWORD` in `.streamlit/secrets.toml` or as an environment variable."
        )
        st.code('APP_PASSWORD = "your-password-here"', language="toml")
        st.stop()

    st.title("🔒 Amazon Warehouse Stock")
    entered = st.text_input("Password", type="password", key="login_password")
    if st.button("Login", type="primary"):
        if hmac.compare_digest(entered, password):
            st.session_state.authenticated = True
            st.rerun()
        st.error("Incorrect password")
    st.stop()


# ── Page ──────────────────────────────────────────────────────────────────────
require_auth()

with st.sidebar:
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()

if "settings" not in st.session_state:
    st.session_state.settings = load_settings()

if "active_snapshot_id" not in st.session_state:
    history = load_history()
    if history:
        st.session_state.active_snapshot_id = history[0]["id"]

settings = st.session_state.settings

st.title("📦 Amazon Warehouse Stock")
st.markdown(
    "Download your **Ledger Report** from "
    "[Seller Central](https://sellercentral.amazon.in/reportcentral/LEDGER_REPORT/1) "
    "and upload the CSV here to see **what stock to send to which city** based on your thresholds."
)
st.markdown(
    "**Date range:** choose **exact dates** only — the **last 2 days, excluding today** "
    "(yesterday and the day before)."
)

uploaded_file = st.file_uploader("📤 Upload Ledger CSV", type=["csv"])

agg = city_agg = velocity = None
source_label = ""
source_key = ""

if uploaded_file is not None and not st.session_state.get("prefer_history"):
    upload_sig = f"{uploaded_file.name}_{uploaded_file.size}"
    source_key = upload_sig
    if st.session_state.get("last_upload_sig") != upload_sig:
        st.session_state.last_upload_sig = upload_sig
        st.session_state.save_decision_for = None
        st.session_state.prefer_history = False
    try:
        df = _read_csv_safe(uploaded_file)
        ledger_data_date = extract_ledger_data_date(df)
        agg, city_agg, velocity = parse_ledger_csv(df)
        source_label = f"Upload · {uploaded_file.name}"
    except Exception as e:
        st.error(str(e))
        st.stop()

    if st.session_state.get("save_decision_for") != upload_sig:
        st.info("Save this upload to history?")
        sc1, sc2 = st.columns(2)
        with sc1:
            if st.button("✅ Yes, save to history", type="primary"):
                snap = append_snapshot(agg, ledger_data_date)
                st.session_state.active_snapshot_id = snap["id"]
                st.session_state.save_decision_for = upload_sig
                st.session_state.prefer_history = False
                source_key = snap["id"]
                st.success(f"Saved snapshot from {snap['uploaded_at']}")
                st.rerun()
        with sc2:
            if st.button("Skip — use without saving"):
                st.session_state.save_decision_for = upload_sig
                st.session_state.active_snapshot_id = None
                st.session_state.prefer_history = False
                st.rerun()

    recorded_date = None
    if st.session_state.get("save_decision_for") == upload_sig:
        saved = get_snapshot(st.session_state.get("active_snapshot_id", ""))
        if saved:
            recorded_date = snapshot_recorded_date(saved)
            ledger_data_date = snapshot_ledger_data_date(saved) or ledger_data_date

    render_app(
        agg, city_agg, velocity, settings, source_label, source_key, recorded_date, ledger_data_date
    )

else:
    snap = None
    active_id = st.session_state.get("active_snapshot_id")
    if active_id:
        snap = get_snapshot(active_id)

    if snap is None:
        history = load_history()
        if history:
            snap = history[0]
            st.session_state.active_snapshot_id = snap["id"]

    if snap:
        agg, city_agg = agg_from_snapshot(snap)
        velocity = None
        source_label = f"History · {snap['uploaded_at']}"
        source_key = snap["id"]
        recorded_date = snapshot_recorded_date(snap)
        ledger_data_date = snapshot_ledger_data_date(snap)
        render_app(
            agg, city_agg, velocity, settings, source_label, source_key, recorded_date, ledger_data_date
        )
    else:
        st.info("Upload a CSV to get started. Saved snapshots appear under **History & Settings**.")
