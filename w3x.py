import base64
import hmac
import json
import os
import re
import urllib.error
import urllib.request
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
LIS_RADIUS_KM = 300
INDIA_MAP_CENTER = [22.0, 79.0]
INDIA_MAP_ZOOM = 5
LIS_CIRCLE_COLORS = (
    "#9b59b6",
    "#e74c3c",
    "#3498db",
    "#2ecc71",
    "#f39c12",
    "#1abc9c",
    "#e67e22",
    "#34495e",
    "#16a085",
    "#c0392b",
    "#8e44ad",
    "#2980b9",
    "#27ae60",
    "#d35400",
    "#7f8c8d",
    "#6c5ce7",
    "#fd79a8",
    "#00b894",
    "#636e72",
    "#e17055",
)


def lis_circle_color(loc: str) -> str:
    return LIS_CIRCLE_COLORS[sum(ord(c) for c in loc) % len(LIS_CIRCLE_COLORS)]

DEFAULT_SETTINGS = {
    "enabled": True,
    "global_threshold": 10,
    "scope": "per_city",
    "sku_overrides": {},
    "selected_cities": [],
    "excluded_warehouses": [],
    "filters_saved": False,
}


def normalize_settings(settings: dict) -> dict:
    merged = {**DEFAULT_SETTINGS, **settings}
    if merged.get("selected_cities") and merged.get("selected_warehouses"):
        merged.pop("selected_warehouses", None)
    return merged


def _read_secret(key: str, default=None):
    try:
        val = st.secrets.get(key, default)
        if val is not None and val != "":
            return val
    except (FileNotFoundError, KeyError, AttributeError):
        pass
    # Support [github] section in secrets.toml
    aliases = {
        "GITHUB_SETTINGS_TOKEN": ("token", "GITHUB_SETTINGS_TOKEN"),
        "GITHUB_SETTINGS_REPO": ("repo", "GITHUB_SETTINGS_REPO"),
        "GITHUB_SETTINGS_PATH": ("path", "GITHUB_SETTINGS_PATH"),
    }
    for section_name in ("github", "GITHUB", "settings"):
        try:
            section = st.secrets.get(section_name)
            if isinstance(section, dict):
                for alias in aliases.get(key, (key,)):
                    if section.get(alias) is not None:
                        return section.get(alias)
        except (FileNotFoundError, KeyError, AttributeError):
            continue
    return default


def github_settings_config() -> dict | None:
    token = _read_secret("GITHUB_SETTINGS_TOKEN")
    repo = _read_secret("GITHUB_SETTINGS_REPO")
    if not token or not repo:
        return None
    return {
        "token": str(token).strip().strip('"').strip("'"),
        "repo": str(repo).strip().strip("/"),
        "path": str(_read_secret("GITHUB_SETTINGS_PATH", "data/settings.json")).strip(),
    }


def _github_api_request(method: str, url: str, token: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "User-Agent": "azstock-settings-sync",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GitHub API {exc.code}: {detail}") from exc


def _github_contents_url(cfg: dict) -> str:
    owner, repo_name = cfg["repo"].split("/", 1)
    return f"https://api.github.com/repos/{owner}/{repo_name}/contents/{cfg['path']}"


def _fetch_github_settings_file(cfg: dict) -> tuple[dict | None, str | None]:
    url = _github_contents_url(cfg)
    try:
        result = _github_api_request("GET", url, cfg["token"])
    except RuntimeError as exc:
        if "GitHub API 404" in str(exc):
            return None, None
        raise
    content = base64.b64decode(result["content"].replace("\n", "")).decode("utf-8")
    return json.loads(content), result.get("sha")


def _merge_session_settings(settings: dict) -> dict:
    """Merge live session settings so partial saves keep overrides and thresholds."""
    merged = dict(st.session_state.get("settings") or {})
    merged.update(settings)
    if "settings_selected_cities" in st.session_state:
        merged["selected_cities"] = list(st.session_state.settings_selected_cities)
    if "settings_excluded_warehouses" in st.session_state:
        merged["excluded_warehouses"] = list(st.session_state.settings_excluded_warehouses)
    if "settings_enabled" in st.session_state:
        merged["enabled"] = bool(st.session_state.settings_enabled)
    if "settings_global_threshold" in st.session_state:
        merged["global_threshold"] = int(st.session_state.settings_global_threshold)
    return merged


def _settings_payload(settings: dict) -> dict:
    merged = _merge_session_settings(settings)
    if merged.get("selected_cities") or merged.get("excluded_warehouses"):
        merged["filters_saved"] = True
    to_save = normalize_settings(merged)
    to_save.pop("selected_warehouses", None)
    # JSON keys must be strings for sku_overrides
    overrides = to_save.get("sku_overrides") or {}
    to_save["sku_overrides"] = {str(k): int(v) for k, v in overrides.items()}
    return to_save


def _write_local_settings(settings: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)


def _load_local_settings_file() -> dict | None:
    if not SETTINGS_FILE.exists():
        return None
    with open(SETTINGS_FILE, encoding="utf-8") as f:
        return normalize_settings(json.load(f))


def _load_github_settings(cfg: dict) -> dict | None:
    raw, sha = _fetch_github_settings_file(cfg)
    if raw is None:
        return None
    st.session_state._settings_github_sha = sha
    return normalize_settings(raw)


def _merge_settings_sources(*sources: dict) -> dict:
    """Merge settings blobs; keep non-empty city, warehouse, and override lists."""
    merged = DEFAULT_SETTINGS.copy()
    for src in sources:
        if not src:
            continue
        norm = normalize_settings(src)
        for key, val in norm.items():
            if key in ("selected_cities", "excluded_warehouses", "sku_overrides"):
                if val:
                    merged[key] = val
            elif key == "filters_saved":
                merged[key] = bool(merged.get("filters_saved")) or bool(val)
            else:
                merged[key] = val
    if merged.get("selected_cities") or merged.get("excluded_warehouses"):
        merged["filters_saved"] = True
    return normalize_settings(merged)


def _load_settings_from_sources() -> dict:
    sources: list[dict] = []
    cfg = github_settings_config()
    if cfg:
        try:
            github_settings = _load_github_settings(cfg)
            if github_settings is not None:
                sources.append(github_settings)
        except Exception as exc:
            st.session_state._settings_load_warning = f"Could not load settings from GitHub: {exc}"

    local_settings = _load_local_settings_file()
    if local_settings is not None:
        sources.append(local_settings)

    if sources:
        return _merge_settings_sources(*sources)

    app_settings = _read_secret("APP_SETTINGS")
    if app_settings:
        if isinstance(app_settings, str):
            return normalize_settings(json.loads(app_settings))
        if isinstance(app_settings, dict):
            return normalize_settings(app_settings)

    return DEFAULT_SETTINGS.copy()


def load_settings(force_refresh: bool = False) -> dict:
    if not force_refresh:
        cached = st.session_state.get("_settings_cache")
        if cached is not None:
            return dict(cached)
    settings = _load_settings_from_sources()
    st.session_state._settings_cache = settings
    return dict(settings)


def refresh_settings_from_storage(
    agg: pd.DataFrame | None = None,
    source_key: str | None = None,
    force: bool = False,
) -> dict:
    """Reload settings.json from disk/GitHub and apply to session + widgets."""
    refresh_key = source_key or "_default"
    if not force and st.session_state.get("_settings_applied_for") == refresh_key:
        return st.session_state.settings

    settings = load_settings(force_refresh=True)
    st.session_state.settings = settings
    apply_filter_widgets_from_settings(settings, agg)
    st.session_state._settings_applied_for = refresh_key
    return settings


def save_settings(settings: dict) -> None:
    to_save = _settings_payload(settings)
    st.session_state._settings_cache = to_save
    st.session_state.settings = to_save

    try:
        _write_local_settings(to_save)
    except OSError as exc:
        st.session_state._settings_save_warning = f"Could not write local settings file: {exc}"

    cfg = github_settings_config()
    if cfg:
        try:
            _push_settings_to_github(cfg, to_save)
            st.session_state.pop("_settings_save_warning", None)
            st.session_state._settings_save_ok = True
        except Exception as exc:
            st.session_state._settings_save_warning = f"Could not save settings to GitHub: {exc}"
            st.session_state._settings_save_ok = False
    else:
        st.session_state._settings_save_ok = True

    st.session_state.pop("_settings_applied_for", None)


def _push_settings_to_github(cfg: dict, to_save: dict) -> None:
    encoded = base64.b64encode(json.dumps(to_save, indent=2).encode("utf-8")).decode("ascii")
    url = _github_contents_url(cfg)

    for attempt in range(2):
        sha = st.session_state.get("_settings_github_sha")
        if not sha:
            _, sha = _fetch_github_settings_file(cfg)
        payload: dict = {
            "message": "Update app settings",
            "content": encoded,
        }
        if sha:
            payload["sha"] = sha
        try:
            result = _github_api_request("PUT", url, cfg["token"], payload)
            st.session_state._settings_github_sha = result.get("content", {}).get("sha")
            return
        except RuntimeError as exc:
            if attempt == 0 and "GitHub API 409" in str(exc):
                st.session_state.pop("_settings_github_sha", None)
                continue
            raise


def send_plan_row_key(row: pd.Series, view_mode: str) -> tuple:
    if view_mode == "City":
        return (str(row["City"]), str(row["MSKU"]))
    return (str(row["City"]), str(row["Warehouse"]), str(row["MSKU"]))


def send_qty_store_key(source_key: str, view_mode: str) -> str:
    return f"send_qty_store_{source_key}_{view_mode.lower()}"


def suggested_send_qty(row: pd.Series) -> int:
    return max(0, int(row["Threshold"]) - int(row["Current"]))


def send_qty_is_blank(val) -> bool:
    return val is None or pd.isna(val)


def sum_send_qty(series: pd.Series) -> float:
    """Sum send qty for city rollup; stay blank when every warehouse row is blank."""
    if series.notna().any():
        return float(series.fillna(0).sum())
    return float("nan")


def merge_send_qty_from_store(table: pd.DataFrame, store: dict, view_mode: str) -> pd.DataFrame:
    if table.empty:
        return table
    out = table.copy()
    if "Send qty" not in out.columns:
        out["Send qty"] = pd.NA
    for idx, row in out.iterrows():
        key = send_plan_row_key(row, view_mode)
        if key in store:
            out.at[idx, "Send qty"] = int(store[key])
    return out


def persist_send_qty_to_store(
    before: pd.DataFrame,
    after: pd.DataFrame,
    store: dict,
    view_mode: str,
) -> None:
    for idx, row in after.iterrows():
        key = send_plan_row_key(row, view_mode)
        val = row.get("Send qty")
        before_val = before.loc[idx, "Send qty"] if idx in before.index else pd.NA

        if send_qty_is_blank(val):
            store.pop(key, None)
        elif int(val) == 0 and send_qty_is_blank(before_val) and key not in store:
            continue
        else:
            store[key] = int(val)


def export_csv_with_metadata(df: pd.DataFrame, metadata: list[str], footer: list[str] | None = None) -> bytes:
    lines = [f"# {line}" for line in metadata]
    body = df.to_csv(index=False)
    if footer:
        padded = (footer + [""] * len(df.columns))[: len(df.columns)]
        footer_df = pd.DataFrame([dict(zip(df.columns, padded))], columns=df.columns)
        body = body.rstrip("\n") + "\n" + footer_df.to_csv(index=False, header=False)
    return ("\n".join(lines) + "\n" + body).encode("utf-8")


def apply_filter_widgets_from_settings(settings: dict, agg: pd.DataFrame | None = None) -> None:
    """Align settings widget keys with persisted settings."""
    city_options = all_city_options(agg)
    saved_cities = settings.get("selected_cities") or []
    if not saved_cities and settings.get("selected_warehouses"):
        saved_cities = sorted({extract_city_code(wh) for wh in settings["selected_warehouses"]})
    default_cities = [city for city in saved_cities if city in city_options]
    st.session_state.settings_selected_cities = default_cities

    warehouse_options = warehouses_for_cities(default_cities or city_options, agg)
    st.session_state.settings_excluded_warehouses = [
        wh for wh in (settings.get("excluded_warehouses") or []) if wh in warehouse_options
    ]
    st.session_state.settings_enabled = bool(settings.get("enabled", True))
    st.session_state.settings_global_threshold = int(settings.get("global_threshold", DEFAULT_SETTINGS["global_threshold"]))


def _ensure_settings() -> dict:
    """Return session settings, loading from disk if needed (on_change runs before main script)."""
    if "settings" not in st.session_state:
        st.session_state.settings = load_settings()
    return dict(st.session_state.settings)


def bootstrap_settings(agg: pd.DataFrame | None = None) -> dict:
    """Load settings from disk/GitHub once per browser session and seed filter widgets."""
    if not st.session_state.get("_settings_bootstrapped"):
        refresh_settings_from_storage(agg, force=True)
        st.session_state._settings_bootstrapped = True
    return st.session_state.settings


def _persist_city_filters(agg: pd.DataFrame | None = None) -> None:
    settings = _ensure_settings()
    cities = list(st.session_state.settings_selected_cities)
    settings["selected_cities"] = cities
    settings["filters_saved"] = True
    wh_opts = set(warehouses_for_cities(cities, agg))
    excluded = list(
        st.session_state.get("settings_excluded_warehouses")
        or settings.get("excluded_warehouses")
        or []
    )
    excluded = [wh for wh in excluded if wh in wh_opts]
    st.session_state.settings_excluded_warehouses = excluded
    settings["excluded_warehouses"] = excluded
    st.session_state.settings = settings
    save_settings(settings)


def _persist_excluded_warehouses(agg: pd.DataFrame | None = None) -> None:
    settings = _ensure_settings()
    cities = settings.get("selected_cities") or list(st.session_state.settings_selected_cities)
    wh_opts = set(warehouses_for_cities(cities, agg))
    excluded = [wh for wh in st.session_state.settings_excluded_warehouses if wh in wh_opts]
    st.session_state.settings_excluded_warehouses = excluded
    settings["excluded_warehouses"] = excluded
    settings["filters_saved"] = True
    st.session_state.settings = settings
    save_settings(settings)


def _persist_alert_settings() -> None:
    settings = _ensure_settings()
    settings["enabled"] = bool(st.session_state.settings_enabled)
    settings["global_threshold"] = int(st.session_state.settings_global_threshold)
    settings["scope"] = "per_city"
    st.session_state.settings = settings
    save_settings(settings)


@st.cache_data
def load_warehouse_locations() -> dict[str, dict]:
    with open(LOCATIONS_FILE, encoding="utf-8") as f:
        return json.load(f)


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


def format_export_date(value) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        parsed = pd.to_datetime(text, dayfirst=True, errors="coerce")
    if pd.isna(parsed):
        return None
    return f"{int(parsed.day)} {parsed.strftime('%B')} {parsed.year}"


def send_plan_export_columns(view_mode: str) -> list[str]:
    if view_mode == "City":
        return ["Priority", "City", "MSKU", "Current", "Send qty", "Notes"]
    return ["Priority", "City", "Warehouse", "MSKU", "Current", "Send qty", "Notes"]


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

    expanded = expand_city_warehouse_stock(agg, settings)
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


def known_warehouses_df() -> pd.DataFrame:
    locations = load_warehouse_locations()
    rows = [
        {"Location": code, "CityCode": extract_city_code(code)}
        for code in sorted(locations.keys())
    ]
    return pd.DataFrame(rows)


def all_city_options(agg: pd.DataFrame | None = None) -> list[str]:
    options = set(known_warehouses_df()["CityCode"].tolist())
    if agg is not None and not agg.empty:
        options.update(agg["CityCode"].astype(str).unique())
    return sorted(options)


def effective_selected_cities(settings: dict, agg: pd.DataFrame | None = None) -> list[str]:
    all_cities = all_city_options(agg)
    if not settings.get("filters_saved"):
        return all_cities
    selected = settings.get("selected_cities")
    if selected is None and settings.get("selected_warehouses"):
        selected = sorted({extract_city_code(wh) for wh in settings["selected_warehouses"]})
    selected = selected or []
    return [city for city in selected if city in all_cities]


def warehouses_for_cities(city_codes: list[str], agg: pd.DataFrame | None = None) -> list[str]:
    allowed = set(city_codes)
    warehouses = set(
        known_warehouses_df()
        .loc[lambda df: df["CityCode"].isin(allowed), "Location"]
        .tolist()
    )
    if agg is not None and not agg.empty:
        warehouses.update(
            agg.loc[agg["CityCode"].isin(allowed), "Location"].astype(str).tolist()
        )
    return sorted(warehouses)


def excluded_warehouses(settings: dict) -> set[str]:
    return set(settings.get("excluded_warehouses") or [])


def effective_warehouses(settings: dict, agg: pd.DataFrame | None = None) -> list[str]:
    cities = effective_selected_cities(settings, agg)
    warehouses = warehouses_for_cities(cities, agg)
    excluded = excluded_warehouses(settings)
    return [loc for loc in warehouses if loc not in excluded]


def recompute_city_agg(agg: pd.DataFrame) -> pd.DataFrame:
    if agg.empty:
        return agg.copy()
    city_parts = {"Ending Warehouse Balance": "sum"}
    if "In Transit Between Warehouses" in agg.columns:
        city_parts["In Transit Between Warehouses"] = "sum"
    return agg.groupby(["CityCode", "MSKU"], as_index=False).agg(city_parts)


def apply_city_filter(
    agg: pd.DataFrame,
    city_agg: pd.DataFrame,
    settings: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cities = effective_selected_cities(settings, agg)
    if not cities:
        empty = agg.iloc[0:0].copy()
        return empty, recompute_city_agg(empty)
    filtered = agg[agg["CityCode"].isin(cities)].copy()
    excluded = excluded_warehouses(settings)
    if excluded:
        filtered = filtered[~filtered["Location"].astype(str).isin(excluded)].copy()
    return filtered, recompute_city_agg(filtered)


def expand_city_warehouse_stock(agg: pd.DataFrame, settings: dict | None = None) -> pd.DataFrame:
    """Selected cities' FCs × all MSKUs present in the upload, zero-filled where absent."""
    stock = (
        agg.groupby(["Location", "CityCode", "MSKU"], as_index=False)["Ending Warehouse Balance"]
        .sum()
    )
    stock["MSKU"] = stock["MSKU"].astype(str)
    csv_warehouses = stock[["Location", "CityCode"]].drop_duplicates()
    warehouses = (
        pd.concat([csv_warehouses, known_warehouses_df()], ignore_index=True)
        .drop_duplicates(subset=["Location"])
        .sort_values(["CityCode", "Location"])
    )
    if settings is not None:
        cities = set(effective_selected_cities(settings, agg))
        warehouses = warehouses[warehouses["CityCode"].isin(cities)]
        excluded = excluded_warehouses(settings)
        if excluded:
            warehouses = warehouses[~warehouses["Location"].isin(excluded)]
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
    expanded = expand_city_warehouse_stock(agg, settings)
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
                "Send qty": pd.NA,
                "_low": current <= threshold,
                "_zero": current == 0,
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
            Send_qty=("Send qty", sum_send_qty),
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


def send_plan_includes_row(row: pd.Series, view_mode: str) -> bool:
    if view_mode == "City":
        return bool(row.get("_low", False))
    if bool(row.get("_zero", False)):
        return True
    return bool(row.get("_low", False))


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


def overview_low_stock_table(plan: pd.DataFrame, zero_only: bool = False) -> pd.DataFrame:
    if plan.empty:
        return plan.copy()
    view = plan[plan["_low"]].copy()
    if zero_only:
        view = view[view["Current"] == 0]
    if view.empty:
        return view
    view["Status"] = view["Current"].apply(lambda q: "ZERO" if q == 0 else "LOW")
    return view[
        ["City", "Warehouse", "MSKU", "Current", "Threshold", "Status"]
    ].sort_values(["Current", "Threshold"], ascending=[True, False]).reset_index(drop=True)


def overview_warehouse_table(plan: pd.DataFrame) -> pd.DataFrame:
    if plan.empty:
        return plan.copy()
    view = plan.copy()
    view["Status"] = view.apply(
        lambda r: "ZERO" if r["Current"] == 0 else ("LOW" if r["_low"] else "OK"),
        axis=1,
    )
    return view[
        ["Warehouse", "City", "MSKU", "Current", "Threshold", "Status"]
    ].sort_values(["Status", "Current"], ascending=[True, True]).reset_index(drop=True)


def city_overview_metrics(city_code: str, send_plan: pd.DataFrame, city_agg: pd.DataFrame) -> dict:
    city_stock = int(
        city_agg.loc[city_agg["CityCode"] == city_code, "Ending Warehouse Balance"].sum()
    )
    city_plan = send_plan[send_plan["CityCode"] == city_code]
    low = city_plan[city_plan["_low"]] if not city_plan.empty else city_plan
    return {
        "total_stock": city_stock,
        "low_lines": len(low),
        "warehouses": int(city_plan["Warehouse"].nunique()) if not city_plan.empty else 0,
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


def _map_alert_table_html(alert_rows: list[dict]) -> str:
    if not alert_rows:
        return ""
    lines = [
        "<table style='font-size:12px;border-collapse:collapse;width:100%'>",
        "<tr><th style='text-align:left;padding:2px 6px'>MSKU</th>"
        "<th style='text-align:right;padding:2px 6px'>Qty</th>"
        "<th style='text-align:right;padding:2px 6px'>Threshold</th>"
        "<th style='padding:2px 6px'>Status</th></tr>",
    ]
    for row in alert_rows:
        qty = int(row["qty"])
        status = "ZERO" if qty == 0 else "LOW"
        lines.append(
            f"<tr><td style='padding:2px 6px'>{row['MSKU']}</td>"
            f"<td style='text-align:right;padding:2px 6px'>{qty:,}</td>"
            f"<td style='text-align:right;padding:2px 6px'>{int(row['threshold']):,}</td>"
            f"<td style='padding:2px 6px'>{status}</td></tr>"
        )
    lines.append("</table>")
    return "".join(lines)


def _alerts_for_location(loc: str, alerts_data: dict) -> list[dict]:
    rows = [
        {
            "MSKU": a["MSKU"],
            "qty": int(a["qty"]),
            "threshold": int(a["threshold"]),
        }
        for a in alerts_data.get("alerts", [])
        if a.get("location") == loc
    ]
    return sorted(rows, key=lambda r: (r["qty"], r["MSKU"]))


def _alerts_for_city(city_code: str, agg: pd.DataFrame, alerts_data: dict) -> list[dict]:
    flagged_mskus = {
        msku for loc, msku in alerts_data.get("flagged_skus", set()) if extract_city_code(loc) == city_code
    }
    if not flagged_mskus:
        return []
    city_totals = {
        str(msku): int(qty)
        for msku, qty in (
            agg[agg["CityCode"] == city_code]
            .groupby("MSKU")["Ending Warehouse Balance"]
            .sum()
            .items()
        )
    }
    threshold_by_msku = {
        str(a["MSKU"]): int(a["threshold"])
        for a in alerts_data.get("alerts", [])
        if extract_city_code(a.get("location", "")) == city_code
    }
    rows = [
        {
            "MSKU": str(msku),
            "qty": city_totals.get(str(msku), 0),
            "threshold": threshold_by_msku.get(str(msku), 0),
        }
        for msku in flagged_mskus
    ]
    return sorted(rows, key=lambda r: (r["qty"], r["MSKU"]))


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


def map_location_totals(agg: pd.DataFrame, settings: dict) -> pd.DataFrame:
    """All warehouses in selected cities (minus exclusions), with zero sellable where absent from upload."""
    warehouses = effective_warehouses(settings, agg)
    totals = _location_totals(agg)
    sellable_by_loc = totals.set_index("Location")["sellable"].to_dict() if not totals.empty else {}
    rows = [
        {
            "Location": loc,
            "CityCode": extract_city_code(loc),
            "sellable": int(sellable_by_loc.get(loc, 0)),
        }
        for loc in warehouses
    ]
    return pd.DataFrame(rows)


def build_warehouse_map(
    agg: pd.DataFrame,
    locations_db: dict,
    mode: str,
    selected_cities: list[str] | None,
    alerts_data: dict | None = None,
    show_low_stock_only: bool = False,
    scope: str = "per_city",
    show_lis_radius: bool = False,
    satellite_view: bool = False,
    settings: dict | None = None,
) -> tuple[folium.Map, list[str]]:
    alerts_data = alerts_data or compute_low_stock_alerts(agg, {"enabled": False})
    loc_totals = map_location_totals(agg, settings) if settings else _location_totals(agg)
    unmapped = [loc for loc in loc_totals["Location"] if loc not in locations_db]

    if satellite_view:
        m = folium.Map(location=INDIA_MAP_CENTER, zoom_start=INDIA_MAP_ZOOM, tiles=None)
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri World Imagery",
            overlay=False,
            control=False,
        ).add_to(m)
        folium.TileLayer(
            "OpenStreetMap",
            overlay=True,
            control=False,
            opacity=0.6,
            show=True,
        ).add_to(m)
    else:
        m = folium.Map(location=INDIA_MAP_CENTER, zoom_start=INDIA_MAP_ZOOM, tiles="OpenStreetMap")

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
            alert_rows = _alerts_for_location(loc, alerts_data)
            alert_table = _map_alert_table_html(alert_rows)
            sku_table = _sku_table_html(
                agg[agg["Location"] == loc].groupby("MSKU", as_index=False)[
                    "Ending Warehouse Balance"
                ].sum(),
                alerts_data,
                loc,
                is_city=False,
                scope=scope,
            )
            alert_block = (
                f"<br><b>⚠️ Low-stock alerts ({len(alert_rows)}):</b><br>{alert_table}"
                if alert_rows
                else ""
            )
            popup_html = (
                f"<b>{loc}</b><br>"
                f"{meta.get('address', '')}<br>"
                f"{meta.get('city', '')}, {meta.get('state', '')}<br>"
                f"<b>Sellable:</b> {sellable:,}"
                f"{alert_block}"
                f"<br><br><b>All SKUs:</b><br>{sku_table}"
            )
            color = "#d9534f" if is_low else "#2c6fad"
            fill = "#e74c3c" if is_low else "#4a90d9"
            radius = 13 if is_low else 10
            tooltip = f"{loc} — {sellable:,} sellable"
            if alert_n:
                tooltip += f" · ⚠️ {alert_n} alert{'s' if alert_n != 1 else ''}"

            if show_lis_radius:
                lis_color = lis_circle_color(loc)
                folium.Circle(
                    location=[lat, lng],
                    radius=LIS_RADIUS_KM * 1000,
                    color=lis_color,
                    fill=True,
                    fill_color=lis_color,
                    fill_opacity=0.12,
                    weight=2,
                    tooltip=f"{loc} — {LIS_RADIUS_KM} km LIS radius",
                ).add_to(m)

            folium.CircleMarker(
                location=[lat, lng],
                radius=radius,
                color=color,
                fill=True,
                fill_color=fill,
                fill_opacity=0.85,
                tooltip=tooltip,
                popup=folium.Popup(popup_html, max_width=440),
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
            alert_rows = _alerts_for_city(city_code, agg, alerts_data)
            alert_table = _map_alert_table_html(alert_rows)

            city_sku = (
                agg[agg["CityCode"] == city_code]
                .groupby("MSKU", as_index=False)["Ending Warehouse Balance"]
                .sum()
            )
            sku_table = _sku_table_html(city_sku, alerts_data, city_code, is_city=True, scope=scope)

            wh_list = ", ".join(sorted(mapped["Location"].astype(str)))
            alert_block = (
                f"<br><b>⚠️ Low-stock alerts ({len(alert_rows)}):</b><br>{alert_table}"
                if alert_rows
                else ""
            )
            popup_html = (
                f"<b>{city_name}</b> ({city_code})<br>"
                f"<b>Warehouses ({fc_count}):</b> {wh_list}<br>"
                f"<b>Sellable:</b> {total_sellable:,}"
                f"{alert_block}"
                f"<br><br><b>All SKUs (city total):</b><br>{sku_table}"
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
                popup=folium.Popup(popup_html, max_width=440),
            ).add_to(m)

    return m, unmapped


def render_settings_panel(settings: dict, agg: pd.DataFrame | None = None) -> dict:
    st.subheader("⚙️ Alert thresholds")
    st.caption("Each warehouse is checked separately. If an MSKU appears anywhere, every FC in selected cities is listed (0 where absent).")

    if "settings_enabled" not in st.session_state:
        st.session_state.settings_enabled = bool(settings.get("enabled", True))
    if "settings_global_threshold" not in st.session_state:
        st.session_state.settings_global_threshold = int(settings.get("global_threshold", 10))

    settings["enabled"] = st.checkbox(
        "Enable low-stock alerts",
        key="settings_enabled",
        on_change=_persist_alert_settings,
    )
    settings["global_threshold"] = st.number_input(
        "Default threshold (all MSKUs)",
        min_value=0,
        key="settings_global_threshold",
        on_change=_persist_alert_settings,
    )
    settings["scope"] = "per_city"

    st.divider()
    st.subheader("🏙️ Cities")
    st.caption("All settings save automatically when changed.")
    cfg = github_settings_config()
    if cfg:
        st.caption(f"Persisting to GitHub: `{cfg['repo']}` → `{cfg['path']}`")
    else:
        st.caption(f"Persisting to local file: `{SETTINGS_FILE.name}` (add GitHub secrets on Streamlit Cloud for durable storage).")
    load_warning = st.session_state.get("_settings_load_warning")
    save_warning = st.session_state.get("_settings_save_warning")
    if load_warning:
        st.warning(load_warning)
    if save_warning:
        st.warning(save_warning)
    elif st.session_state.pop("_settings_save_ok", False):
        st.success("Settings saved.")
    city_options = all_city_options(agg)
    settings["selected_cities"] = st.multiselect(
        "Cities to include",
        options=city_options,
        format_func=lambda code: f"{city_display_name(code)} ({code})",
        help="All warehouses in a selected city are included.",
        key="settings_selected_cities",
        on_change=_persist_city_filters,
        kwargs={"agg": agg},
    )
    if settings["selected_cities"]:
        enabled_wh = effective_warehouses(settings, agg)
        excluded = excluded_warehouses(settings)
        if excluded:
            st.caption(
                f"**{len(enabled_wh)}** warehouses enabled "
                f"({len(excluded)} excluded: {', '.join(sorted(excluded))})"
            )
        else:
            st.caption(f"**{len(enabled_wh)}** warehouses enabled: {', '.join(enabled_wh)}")
    else:
        st.warning("Select at least one city.")

    st.divider()
    st.subheader("🏭 Warehouses")
    warehouse_options = warehouses_for_cities(settings["selected_cities"] or [], agg)
    settings["excluded_warehouses"] = st.multiselect(
        "Excluded warehouses",
        options=warehouse_options,
        help="These FCs are omitted from Send Plan, Overview, and Map.",
        key="settings_excluded_warehouses",
        on_change=_persist_excluded_warehouses,
        kwargs={"agg": agg},
    )
    settings["selected_cities"] = list(st.session_state.settings_selected_cities)
    settings["excluded_warehouses"] = list(st.session_state.settings_excluded_warehouses)
    st.session_state.settings["selected_cities"] = settings["selected_cities"]
    st.session_state.settings["excluded_warehouses"] = settings["excluded_warehouses"]

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
                overrides[str(pick_msku)] = int(st.session_state.settings_pick_thresh)
                settings["sku_overrides"] = overrides
                st.session_state.settings = settings
                save_settings(settings)
                st.rerun()

        if overrides:
            for msku in sorted(overrides):
                rc1, rc2 = st.columns([4, 1])
                rc1.markdown(f"**{msku}** → {overrides[msku]} units")
                if rc2.button("Remove", key=f"settings_rm_{msku}"):
                    del overrides[msku]
                    settings["sku_overrides"] = overrides
                    st.session_state.settings = settings
                    save_settings(settings)
                    st.rerun()
            if st.button("Clear all overrides"):
                settings["sku_overrides"] = {}
                st.session_state.settings = settings
                save_settings(settings)
                st.rerun()
        else:
            st.caption("No overrides — all MSKUs use the default threshold.")

    sc1, sc2 = st.columns(2)
    with sc1:
        if st.button("Save settings", type="primary"):
            settings["enabled"] = bool(st.session_state.settings_enabled)
            settings["global_threshold"] = int(st.session_state.settings_global_threshold)
            settings["selected_cities"] = list(st.session_state.settings_selected_cities)
            settings["excluded_warehouses"] = list(st.session_state.settings_excluded_warehouses)
            wh_opts = set(warehouses_for_cities(settings.get("selected_cities") or [], agg))
            settings["excluded_warehouses"] = [
                wh for wh in (settings.get("excluded_warehouses") or []) if wh in wh_opts
            ]
            settings = normalize_settings(settings)
            st.session_state.settings = settings
            save_settings(settings)
            st.success("Settings saved.")
            st.rerun()
    with sc2:
        if st.button("Reset to defaults"):
            settings = DEFAULT_SETTINGS.copy()
            st.session_state.settings = settings
            st.session_state._settings_cache = settings
            apply_filter_widgets_from_settings(settings, agg)
            save_settings(settings)
            st.rerun()

    st.session_state.settings = _merge_session_settings(settings)
    return st.session_state.settings


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
        ledger = snapshot_ledger_data_date(snap)
        label = f"{'▶ ' if is_active else ''}{snap['uploaded_at']}"
        if ledger:
            label += f" · Ledger: {ledger}"
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
            f"**{summary['skus']}** lines below threshold"
        )


def prepare_send_plan_table(
    plan: pd.DataFrame,
    selected_skus: list[str],
    view_mode: str,
    show_all: bool,
    sort_by: str,
    sort_asc: bool,
) -> tuple[pd.DataFrame, list[str]]:
    view = plan[plan["MSKU"].astype(str).isin(selected_skus)].copy()

    if view_mode == "City":
        view = aggregate_send_plan_by_city(view)
        display_cols = ["Priority", "City", "MSKU", "Current", "Threshold", "Suggested", "Send qty"]
    else:
        display_cols = [
            "Priority", "City", "Warehouse", "MSKU", "Current", "Threshold", "Suggested", "Send qty"
        ]

    if not show_all:
        view = view[view.apply(lambda r: send_plan_includes_row(r, view_mode), axis=1)]

    if view.empty:
        return view, display_cols

    view["Suggested"] = view.apply(suggested_send_qty, axis=1)
    if "Send qty" not in view.columns:
        view["Send qty"] = pd.NA

    if sort_by not in display_cols:
        sort_by = "Priority"
    view = view.sort_values(sort_by, ascending=sort_asc, kind="mergesort").reset_index(drop=True)
    view["Priority"] = range(1, len(view) + 1)
    return view[display_cols].copy(), display_cols


def render_send_plan_tab(
    plan: pd.DataFrame,
    source_key: str,
    ledger_data_date: str | None = None,
) -> None:
    st.subheader("📋 Send Plan")
    st.caption("Review low-stock lines and enter how many units you plan to send. Export matches the table row order below.")

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
        show_all = st.checkbox(
            "Show all SKUs (including above threshold)",
            value=False,
            key=f"pref_send_plan_show_all_{source_key}",
        )

    if not selected_skus:
        st.warning("Select at least one MSKU to include in the send plan.")
        return

    display_cols = (
        ["Priority", "City", "MSKU", "Current", "Threshold", "Suggested", "Send qty"]
        if view_mode == "City"
        else [
            "Priority", "City", "Warehouse", "MSKU", "Current", "Threshold", "Suggested", "Send qty"
        ]
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

    store_key = send_qty_store_key(source_key, view_mode)
    qty_store: dict = st.session_state.setdefault(store_key, {})
    table = merge_send_qty_from_store(table, qty_store, view_mode)
    table["Send qty"] = pd.to_numeric(table["Send qty"], errors="coerce")

    editor_key = f"send_plan_editor_{view_mode.lower()}_{source_key}"
    fill1, fill2, fill3, _ = st.columns([1, 1, 1, 2])
    with fill1:
        if st.button("Fill send qty with gap", key=f"send_fill_gap_{view_mode}_{source_key}"):
            for _, row in table.iterrows():
                qty_store[send_plan_row_key(row, view_mode)] = suggested_send_qty(row)
            st.session_state[store_key] = qty_store
            st.session_state.pop(editor_key, None)
            st.rerun()
    with fill2:
        if st.button("Fill zeros only", key=f"send_fill_zero_{view_mode}_{source_key}"):
            for _, row in table.iterrows():
                if int(row["Current"]) == 0:
                    qty_store[send_plan_row_key(row, view_mode)] = suggested_send_qty(row)
            st.session_state[store_key] = qty_store
            st.session_state.pop(editor_key, None)
            st.rerun()
    with fill3:
        if st.button("Clear send quantities", key=f"send_clear_{view_mode}_{source_key}"):
            st.session_state[store_key] = {}
            st.session_state.pop(editor_key, None)
            st.rerun()

    column_config = {
        "Priority": st.column_config.NumberColumn("Priority", disabled=True),
        "City": st.column_config.TextColumn("City", disabled=True),
        "MSKU": st.column_config.TextColumn("MSKU", disabled=True),
        "Current": st.column_config.NumberColumn("Current", disabled=True),
        "Threshold": st.column_config.NumberColumn("Threshold", disabled=True),
        "Suggested": st.column_config.NumberColumn("Suggested", disabled=True),
        "Send qty": st.column_config.NumberColumn("Send qty", min_value=0, step=1),
    }
    if view_mode == "Warehouse":
        column_config["Warehouse"] = st.column_config.TextColumn("Warehouse", disabled=True)

    edited = st.data_editor(
        table,
        column_config=column_config,
        hide_index=True,
        use_container_width=True,
        key=editor_key,
    )
    persist_send_qty_to_store(table, edited, qty_store, view_mode)
    st.session_state[store_key] = qty_store

    total_send = int(edited["Send qty"].dropna().sum())
    st.caption(f"**Total send qty:** {total_send:,} units")

    export_cols = send_plan_export_columns(view_mode)
    export_df = edited[[c for c in export_cols if c != "Notes"]].copy()
    export_df["Send qty"] = export_df["Send qty"].apply(
        lambda x: "" if send_qty_is_blank(x) else int(x)
    )
    export_df["Notes"] = ""

    export_date = format_export_date(ledger_data_date) or format_export_date(datetime.now())
    export_meta = [
        f"Exported: {format_export_date(datetime.now())}",
        f"View: {view_mode}",
        f"SKUs: {', '.join(selected_skus)}",
    ]
    if export_date:
        export_meta.append(f"Ledger data date: {export_date}")

    footer = (
        ["TOTAL", "", "", "", str(total_send), ""]
        if view_mode == "City"
        else ["TOTAL", "", "", "", "", str(total_send), ""]
    )

    date_slug = export_date.replace(" ", "_") if export_date else datetime.now().strftime("%d_%B_%Y")
    st.download_button(
        label="⬇️ Export send plan CSV",
        data=export_csv_with_metadata(export_df[export_cols], export_meta, footer=footer),
        file_name=f"send_plan_{view_mode.lower()}_{date_slug}.csv",
        mime="text/csv",
    )


def render_overview_tab(
    agg: pd.DataFrame,
    city_agg: pd.DataFrame,
    velocity: pd.DataFrame | None,
    settings: dict,
    send_plan: pd.DataFrame,
    source_key: str,
) -> None:
    all_skus = sorted(send_plan["MSKU"].astype(str).unique()) if not send_plan.empty else []
    selected_skus = st.multiselect(
        "SKUs to include",
        options=all_skus,
        default=all_skus,
        help="Exclude SKUs you don't want shown in this overview.",
        key=f"overview_skus_{source_key}",
    )
    if not selected_skus:
        st.warning("Select at least one MSKU to include in the overview.")
        return

    sku_filter = send_plan["MSKU"].astype(str).isin(selected_skus)
    plan = send_plan[sku_filter].copy()
    filtered_agg = agg[agg["MSKU"].astype(str).isin(selected_skus)].copy()
    filtered_city_agg = city_agg[city_agg["MSKU"].astype(str).isin(selected_skus)].copy()
    filtered_velocity = (
        velocity[velocity["MSKU"].astype(str).isin(selected_skus)].copy()
        if velocity is not None and not velocity.empty
        else velocity
    )

    total_in_transit = (
        int(filtered_agg["In Transit Between Warehouses"].sum())
        if "In Transit Between Warehouses" in filtered_agg.columns
        else 0
    )
    total_cities = int(filtered_city_agg["CityCode"].nunique()) if not filtered_city_agg.empty else 0
    total_warehouses = int(filtered_agg["Location"].nunique()) if not filtered_agg.empty else 0
    total_sellable = int(filtered_agg["Ending Warehouse Balance"].sum()) if not filtered_agg.empty else 0
    summary = send_plan_summary(plan)
    flagged_cities = (
        set(plan.loc[plan["_low"], "CityCode"].unique()) if not plan.empty and "_low" in plan.columns else set()
    )

    st.subheader("📊 Stock health")
    if settings.get("enabled") and summary["skus"] > 0:
        st.warning(
            f"**{summary['cities']}** of **{total_cities}** cities · "
            f"**{summary['warehouses']}** of **{total_warehouses}** warehouses · "
            f"**{summary['skus']}** low-stock lines"
        )
    else:
        st.success("All warehouse × MSKU lines are above threshold.")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total sellable", total_sellable)
    k2.metric("In transit", total_in_transit)
    k3.metric("Cities flagged", summary["cities"])
    k4.metric("Warehouses flagged", summary["warehouses"])
    k5.metric("Low-stock lines", summary["skus"])

    if settings.get("enabled") and not plan.empty:
        st.subheader("⚠️ Low stock alerts")
        fc1, fc2 = st.columns([2, 1])
        with fc1:
            alert_group = st.radio(
                "Group by",
                ["Flat list", "City", "Warehouse"],
                horizontal=True,
                key=f"overview_alert_group_{source_key}",
            )
        with fc2:
            zero_only = st.checkbox("Zero stock only", value=False, key=f"overview_zero_only_{source_key}")

        alerts_df = overview_low_stock_table(plan, zero_only=zero_only)
        if alerts_df.empty:
            st.info("No alerts match your filters.")
        elif alert_group == "Flat list":
            st.dataframe(alerts_df, use_container_width=True, hide_index=True)
            st.download_button(
                "⬇️ Export alerts CSV",
                data=export_csv_with_metadata(
                    alerts_df,
                    [
                        f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                        "Section: Low stock alerts",
                    ],
                ),
                file_name=f"alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key=f"export_alerts_{source_key}",
            )
        elif alert_group == "City":
            for city_code in sorted(alerts_df["City"].unique()):
                city_rows = alerts_df[alerts_df["City"] == city_code]
                with st.expander(f"{city_rows.iloc[0]['City']} — {len(city_rows)} alerts", expanded=False):
                    st.dataframe(city_rows.drop(columns=["City"]), use_container_width=True, hide_index=True)
        else:
            for wh in sorted(alerts_df["Warehouse"].unique()):
                wh_rows = alerts_df[alerts_df["Warehouse"] == wh]
                with st.expander(f"{wh} — {len(wh_rows)} alerts", expanded=False):
                    st.dataframe(wh_rows, use_container_width=True, hide_index=True)

    st.subheader("🏙️ Stock by city")
    if filtered_city_agg.empty:
        st.info("No city stock for the selected SKUs.")
    else:
        city_order = (
            filtered_city_agg.groupby("CityCode")["Ending Warehouse Balance"]
            .sum()
            .sort_values(ascending=False)
        )
        sorted_cities = list(flagged_cities) + [c for c in city_order.index if c not in flagged_cities]
        city_plan = aggregate_send_plan_by_city(plan) if not plan.empty else plan

        for city_chunk in chunks(sorted_cities, 3):
            cols = st.columns(3)
            for i, city_code in enumerate(city_chunk):
                with cols[i]:
                    metrics = city_overview_metrics(city_code, plan, filtered_city_agg)
                    city_warn = " ⚠️" if city_code in flagged_cities else ""
                    label = (
                        f"{city_display_name(city_code)}{city_warn} — "
                        f"{metrics['total_stock']} units · "
                        f"{metrics['low_lines']} low"
                    )
                    with st.expander(label, expanded=city_code in flagged_cities):
                        if not city_plan.empty:
                            cp = city_plan[city_plan["CityCode"] == city_code].copy()
                            cp["Status"] = cp.apply(
                                lambda r: "ZERO" if r["Current"] == 0 else ("LOW" if r["_low"] else "OK"),
                                axis=1,
                            )
                            st.markdown("**By MSKU (city total)**")
                            st.dataframe(
                                cp[["MSKU", "Current", "Threshold", "Status"]],
                                use_container_width=True,
                                hide_index=True,
                            )
                        wh_rows = plan[plan["CityCode"] == city_code].copy()
                        if not wh_rows.empty:
                            wh_rows["Status"] = wh_rows.apply(
                                lambda r: "ZERO" if r["Current"] == 0 else ("LOW" if r["_low"] else "OK"),
                                axis=1,
                            )
                            st.markdown("**By warehouse**")
                            st.dataframe(
                                wh_rows[["Warehouse", "MSKU", "Current", "Threshold", "Status"]],
                                use_container_width=True,
                                hide_index=True,
                            )

    st.subheader("🏬 Stock by warehouse")
    warehouse_df = overview_warehouse_table(plan)
    if warehouse_df.empty:
        st.info("No warehouse data available for the selected SKUs.")
    else:
        st.dataframe(warehouse_df, use_container_width=True, hide_index=True)

    if filtered_velocity is not None and not filtered_velocity.empty:
        with st.expander("📈 Units sold this period"):
            st.dataframe(filtered_velocity.sort_values("Units Sold", ascending=False), hide_index=True)

    st.download_button(
        "⬇️ Download aggregated CSV",
        data=export_csv_with_metadata(
            filtered_agg,
            [f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", "Section: Aggregated sellable by location"],
        ),
        file_name="aggregated_sellable_by_location_msku.csv",
        mime="text/csv",
    )


def render_sidebar_context(
    source_label: str,
    settings: dict,
    agg: pd.DataFrame | None,
    recorded_date: str | None,
    ledger_data_date: str | None,
) -> None:
    st.divider()
    st.caption("At a glance")
    st.markdown(f"**Source:** {source_label}")
    if recorded_date:
        st.markdown(f"**Recorded:** {recorded_date}")
    if ledger_data_date:
        st.markdown(f"**Ledger date:** {ledger_data_date}")
    cities = effective_selected_cities(settings, agg)
    enabled = effective_warehouses(settings, agg)
    excluded = sorted(excluded_warehouses(settings))
    city_labels = ", ".join(f"{city_display_name(c)} ({c})" for c in cities) or "none"
    warehouse_labels = ", ".join(enabled) or "none"
    excluded_labels = ", ".join(excluded) or "none"

    st.markdown(f"**Cities enabled:** {city_labels}")
    st.markdown(f"**Warehouses enabled:** {warehouse_labels}")
    st.markdown(f"**Excluded:** {excluded_labels}")
    if st.button("Open settings", key="sidebar_open_settings"):
        st.session_state.jump_to_settings = True
        st.rerun()


def render_map_tab(agg: pd.DataFrame, settings: dict, alerts_data: dict) -> None:
    st.subheader("🗺️ Warehouse map")
    locations_db = load_warehouse_locations()
    city_options = effective_selected_cities(settings, agg)
    city_labels = {c: city_display_name(c) for c in city_options}

    mc1, mc2, mc3 = st.columns([1, 2, 1])
    with mc1:
        map_mode = st.radio(
            "View",
            ["By city", "All warehouses"],
            horizontal=True,
            key="pref_map_mode",
        )
    with mc2:
        selected_cities = st.multiselect(
            "Cities",
            city_options,
            default=city_options,
            format_func=lambda c: f"{city_labels[c]} ({c})",
            help="Limited to cities enabled in History & Settings.",
            key="pref_map_cities",
        )
    with mc3:
        show_low_only = st.checkbox("Low stock only", value=True, key="pref_map_low_stock_only")

    opt1, opt2 = st.columns(2)
    with opt1:
        satellite_view = st.checkbox("Satellite view", value=False, key="pref_satellite_view")
    with opt2:
        show_lis_radius = False
        if map_mode == "All warehouses":
            show_lis_radius = st.checkbox("SHOW LIS RADIUS", value=False, key="pref_show_lis_radius")

    warehouse_map, unmapped = build_warehouse_map(
        agg,
        locations_db,
        map_mode,
        selected_cities or None,
        alerts_data=alerts_data,
        show_low_stock_only=show_low_only,
        scope="per_city",
        show_lis_radius=show_lis_radius,
        satellite_view=satellite_view,
        settings=settings,
    )
    excluded = excluded_warehouses(settings)
    if excluded:
        st.caption(f"Excluded from map: {', '.join(sorted(excluded))}")
    if unmapped:
        st.warning(f"Unmapped warehouses (no coordinates): {', '.join(unmapped)}")
    if selected_cities:
        st_folium(warehouse_map, width="100%", height=750, returned_objects=[])
    elif not selected_cities:
        st.info("Select at least one city to display the map.")


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
    with st.sidebar:
        render_sidebar_context(source_label, settings, agg, recorded_date, ledger_data_date)

    if st.session_state.pop("jump_to_settings", False):
        st.info("Open the **History & Settings** tab below to change city and warehouse filters.")

    render_data_dates_banner(recorded_date, ledger_data_date)

    st.caption(f"Data source: **{source_label}**")

    # Apply persisted city/warehouse filters from storage for this dataset.
    refresh_settings_from_storage(
        agg,
        source_key,
        force=not st.session_state.settings.get("filters_saved"),
    )
    settings = st.session_state.settings
    settings["scope"] = "per_city"
    agg, city_agg = apply_city_filter(agg, city_agg, settings)
    if agg.empty:
        cities = effective_selected_cities(settings, agg)
        enabled = effective_warehouses(settings, agg)
        excluded = excluded_warehouses(settings)
        st.warning("No data for the current city and warehouse filters.")
        st.markdown(
            f"**Cities enabled:** {', '.join(cities) or 'none'}  \n"
            f"**Warehouses enabled:** {', '.join(enabled) or 'none'}"
        )
        if excluded:
            st.markdown(f"**Excluded FCs:** {', '.join(sorted(excluded))}")
        if st.button("Open settings", key="empty_open_settings"):
            st.session_state.jump_to_settings = True
            st.rerun()
        tab_history = st.tabs(["⚙️ History & Settings"])[0]
        with tab_history:
            render_history_panel()
            st.divider()
            settings = render_settings_panel(settings, agg)
        return

    alerts_data = compute_low_stock_alerts(agg, settings)
    send_plan = build_send_plan(agg, settings)

    render_summary_banner(send_plan)

    tab_send, tab_overview, tab_map, tab_history = st.tabs(
        ["📋 Send Plan", "📊 Overview", "🗺️ Map", "⚙️ History & Settings"]
    )

    with tab_send:
        render_send_plan_tab(send_plan, source_key, ledger_data_date)

    with tab_overview:
        render_overview_tab(agg, city_agg, velocity, settings, send_plan, source_key)

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

bootstrap_settings(agg=None)

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
        st.session_state.pop("_settings_applied_for", None)
        st.session_state.pop("_settings_cache", None)
    try:
        with st.spinner("Parsing ledger…"):
            df = _read_csv_safe(uploaded_file)
            ledger_data_date = extract_ledger_data_date(df)
            agg, city_agg, velocity = parse_ledger_csv(df)
        source_label = f"Upload · {uploaded_file.name}"
        refresh_settings_from_storage(agg, upload_sig, force=True)
    except Exception as e:
        st.error(str(e))
        if "missing required columns" in str(e).lower():
            st.info(
                "Download the report from "
                "[Seller Central Ledger](https://sellercentral.amazon.in/reportcentral/LEDGER_REPORT/1) "
                "and ensure it includes MSKU, Disposition, Ending Warehouse Balance, and Location columns."
            )
        st.stop()

    save_pref = st.session_state.get("upload_save_pref")
    if st.session_state.get("save_decision_for") != upload_sig:
        if save_pref == "always":
            snap = append_snapshot(agg, ledger_data_date)
            st.session_state.active_snapshot_id = snap["id"]
            st.session_state.save_decision_for = upload_sig
            st.session_state.prefer_history = False
            source_key = snap["id"]
        elif save_pref == "never":
            st.session_state.save_decision_for = upload_sig
            st.session_state.active_snapshot_id = None
            st.session_state.prefer_history = False
        else:
            st.info("Save this upload to history?")
            remember = st.checkbox("Remember my choice", key=f"remember_save_{upload_sig}")
            sc1, sc2 = st.columns(2)
            with sc1:
                if st.button("✅ Yes, save to history", type="primary"):
                    if remember:
                        st.session_state.upload_save_pref = "always"
                    snap = append_snapshot(agg, ledger_data_date)
                    st.session_state.active_snapshot_id = snap["id"]
                    st.session_state.save_decision_for = upload_sig
                    st.session_state.prefer_history = False
                    source_key = snap["id"]
                    st.success(f"Saved snapshot from {snap['uploaded_at']}")
                    st.rerun()
            with sc2:
                if st.button("Skip — use without saving"):
                    if remember:
                        st.session_state.upload_save_pref = "never"
                    st.session_state.save_decision_for = upload_sig
                    st.session_state.active_snapshot_id = None
                    st.session_state.prefer_history = False
                    st.rerun()
            st.stop()

    recorded_date = None
    if st.session_state.get("save_decision_for") == upload_sig:
        saved = get_snapshot(st.session_state.get("active_snapshot_id", ""))
        if saved:
            recorded_date = snapshot_recorded_date(saved)
            ledger_data_date = snapshot_ledger_data_date(saved) or ledger_data_date

    render_app(
        agg, city_agg, velocity, st.session_state.settings, source_label, source_key, recorded_date, ledger_data_date
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
            agg, city_agg, velocity, st.session_state.settings, source_label, source_key, recorded_date, ledger_data_date
        )
    else:
        st.info("Upload a CSV to get started. Saved snapshots appear under **History & Settings**.")
