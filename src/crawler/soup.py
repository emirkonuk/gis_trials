import sys
import os
import json
import re
import math
from bs4 import BeautifulSoup, Tag
from typing import Dict, Any, Optional, Tuple, Union

def _sanitize_ws(x):
    if isinstance(x, str):
        return x.replace("\u00A0", " ").replace("\u202F", " ").strip()
    if isinstance(x, list):
        return [_sanitize_ws(v) for v in x]
    if isinstance(x, dict):
        return {k: _sanitize_ws(v) for k, v in x.items()}
    return x

# --- HELPER FUNCTIONS START ---

def _extract_text_in_parentheses(text: Optional[str]) -> Optional[str]:
    """
    Extracts the content from the first pair of parentheses in a string.
    e.g., "Ungefär 750 m till vatten (Mälaren)" -> "Mälaren"
    """
    if not text:
        return None
    match = re.search(r'\((.*?)\)', text)
    return match.group(1).strip() if match else None

def _get_best_value(raw_data: Dict, cat: str, key: str) -> Any:
    """
    Safely gets the best available value from the raw schema.
    It prioritizes json_value, then falls back to html_value.
    """
    val = raw_data.get(cat, {}).get(key, {})
    
    # 1. Prioritize json_value if it's not None
    json_val = val.get("json_value")
    if json_val is not None:
        return json_val
        
    # 2. Fall back to html_value
    html_val = val.get("html_value")
    
    # 3. Handle special case for lat/lon where html_value is a dict
    if isinstance(html_val, dict):
        return html_val.get("button") or html_val.get("offset")
        
    return html_val

def _parse_numeric(value: Any) -> Optional[Union[int, float]]:
    """
    Parses a string like "4 200 000 kr" or "Ungefär 750 m" into a number.
    """
    if isinstance(value, (int, float)):
        return value
    if not isinstance(value, str):
        return None
    
    s = value.replace("\u00A0", "").replace(" ", "")
    s = s.split("kr")[0].split("m²")[0].split("%")[0].split("st")[0].strip()
    s = s.replace(",", ".")
    
    # Find the first valid number in the string
    match = re.search(r'(-?\d+(?:\.\d+)?)', s)
    if not match:
        return None
        
    try:
        num = float(match.group(1))
        # Return int if it's a whole number, else float
        return int(num) if num.is_integer() else num
    except (ValueError, TypeError):
        return None

def _parse_boolean(value: Any) -> Optional[bool]:
    """
    Parses a string like "Ja" or "Nej" into a boolean.
    """
    if isinstance(value, bool):
        return value
    if not isinstance(value, str):
        return None
        
    s = value.strip().lower()
    if s in ('ja', 'true', '1', 'yes', 'finns'):
        return True
    if s in ('nej', 'false', '0', 'no', 'finns ej'):
        return False
        
    return None

def _parse_floor_string(value: Any) -> Dict[str, Any]:
    """
    Parses a complex floor string like "3 av 4, hiss finns"
    into a dictionary of components.
    """
    output = {
        "floor_string": None,
        "floor_number": None,
        "total_floors": None,
        "has_elevator_from_string": None
    }
    if not isinstance(value, str) or not value.strip():
        return output
        
    s = value.strip()
    output["floor_string"] = s
    
    # 1. Check for elevator
    if re.search(r'hiss finns(?! ej)', s, re.IGNORECASE):
        output["has_elevator_from_string"] = True
    elif re.search(r'hiss finns ej', s, re.IGNORECASE):
        output["has_elevator_from_string"] = False
        
    # 2. Check for "X av Y" (e.g., 3 av 4)
    floor_match = re.search(r'(\d+)\s*av\s*(\d+)', s)
    if floor_match:
        try:
            output["floor_number"] = int(floor_match.group(1))
            output["total_floors"] = int(floor_match.group(2))
        except ValueError:
            pass # Keep them None
            
    # 3. Check for simple floor number (e.g., "3" or "3 tr")
    elif not output["floor_number"]:
        simple_match = re.search(r'^(\d+)', s)
        if simple_match:
            try:
                output["floor_number"] = int(simple_match.group(1))
            except ValueError:
                pass

    # 4. Check for special names
    s_lower = s.lower()
    if "bottenvåning" in s_lower or "bv" in s_lower:
        output["floor_number"] = 0
    elif "källarvning" in s_lower or "kv" in s_lower:
        output["floor_number"] = -1
        
    return output

# Master map for (Category, SwedishKey) -> (EnglishKey, ParseType)
# ParseType can be 'numeric', 'bool', 'string', or 'special_floor'
KEY_MAP = {
    # Adress & Plats
    ("Adress & Plats", "gatuadress"): ("street_address", "string"),
    ("Adress & Plats", "område"): ("area", "string"),
    ("Adress & Plats", "kommun"): ("municipality", "string"),
    ("Adress & Plats", "latitud"): ("latitude", "numeric"),
    ("Adress & Plats", "longitud"): ("longitude", "numeric"),
    ("Adress & Plats", "avstånd_vatten_m"): ("distance_to_water_meters", "numeric"),
    ("Adress & Plats", "vatten_namn"): ("water_name", "string"),
    ("Adress & Plats", "avstånd_hav_m"): ("distance_to_ocean_meters", "numeric"),
    ("Adress & Plats", "hav_namn"): ("sea_name", "string"),

    # Priser & Avgifter
    ("Priser & Avgifter", "utgångspris_sek"): ("asking_price_sek", "numeric"),
    ("Priser & Avgifter", "utgångspris_original_sek"): ("original_asking_price_sek", "numeric"), # <-- NEW
    ("Priser & Avgifter", "utgångspris_forandring_procent"): ("price_change_percent", "numeric"), # <-- NEW
    ("Priser & Avgifter", "avgift_månad_sek"): ("fee_per_month_sek", "numeric"),
    ("Priser & Avgifter", "driftkostnad_ar_sek"): ("operating_cost_per_year_sek", "numeric"),
    ("Priser & Avgifter", "minsta_kontantinsats_sek"): ("min_down_payment_sek", "numeric"),
    ("Priser & Avgifter", "utgångspris_per_kvadratmeter_sek"): ("price_per_sqm_sek", "numeric"),

    # Områdesstatistik
    ("Områdesstatistik", "kvadratmeterpris_snitt_sek"): ("area_avg_price_per_sqm_sek", "numeric"),
    ("Områdesstatistik", "prisutveckling_procent"): ("area_price_development_percent", "numeric"),

    # Detaljer & Fastighet
    ("Detaljer & Fastighet", "bostadstyp"): ("property_type", "string"),
    ("Detaljer & Fastighet", "upplåtelseform"): ("tenure", "string"),
    ("Detaljer & Fastighet", "antal_rum"): ("number_of_rooms", "numeric"),
    ("Detaljer & Fastighet", "boarea_kvm"): ("living_area_sqm", "numeric"),
    ("Detaljer & Fastighet", "biarea_kvm"): ("supplemental_area_sqm", "numeric"),
    ("Detaljer & Fastighet", "tomtarea_kvm"): ("land_area_sqm", "numeric"),
    ("Detaljer & Fastighet", "byggår"): ("construction_year", "numeric"),
    ("Detaljer & Fastighet", "våning"): ("floor_info", "special_floor"), # Special handler
    ("Detaljer & Fastighet", "har_balkong"): ("has_balcony", "bool"),
    ("Detaljer & Fastighet", "har_uteplats"): ("has_patio", "bool"),
    ("Detaljer & Fastighet", "har_hiss"): ("has_elevator", "bool"), # Dedicated field
    ("Detaljer & Fastighet", "energiklass"): ("energy_class", "string"),
    ("Detaljer & Fastighet", "planerat_tillträde"): ("planned_access_date", "string"),
    ("Detaljer & Fastighet", "forening_namn"): ("housing_cooperative_name", "string"),

    # Förening
    ("Förening", "ekonomi_status"): ("coop_economy_status", "string"),
    ("Förening", "brf_status"): ("coop_status", "string"),
    ("Förening", "brf_ager_marken"): ("coop_owns_land", "string"), # Often 'Ja'/'Nej' but could be other
    ("Förening", "brf_antal_lagenheter"): ("coop_apartment_count", "numeric"),
    ("Förening", "brf_registreringsar"): ("coop_registration_year", "numeric"),
    ("Förening", "brf_arsavgift"): ("coop_annual_fee_per_sqm", "string"), # e.g. "492 kr/m²"
    ("Förening", "brf_belaning"): ("coop_debt_per_sqm", "string"), # e.g. "5 131 kr/m²"
    ("Förening", "brf_arsredovisning"): ("coop_annual_report_year", "string"), # e.g. "2022"

    # Beskrivning
    ("Beskrivning", "beskrivning_kort"): ("description_short", "string"),
    ("Beskrivning", "beskrivning_detaljerad"): ("description_detailed", "string"),
}

def _clean_and_flatten_data(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Takes the raw, nested, Swedish-keyed dictionary from extract_property_data
    and converts it into a flat, clean, English-keyed dictionary.
    """
    clean_data = {}
    
    for (category, swedish_key), (english_key, parse_type) in KEY_MAP.items():
        raw_value = _get_best_value(raw_data, category, swedish_key)
        
        # Keep the key, but set value to None if raw_value is None or empty string
        if raw_value is None or raw_value == "":
            clean_data[english_key] = None
            if parse_type == "special_floor":
                 # Ensure floor keys are still created as None
                clean_data["floor_string"] = None
                clean_data["floor_number"] = None
                clean_data["total_floors"] = None
                clean_data["has_elevator_from_string"] = None
            continue
            
        try:
            if parse_type == "numeric":
                clean_data[english_key] = _parse_numeric(raw_value)
            elif parse_type == "bool":
                clean_data[english_key] = _parse_boolean(raw_value)
            elif parse_type == "string":
                clean_data[english_key] = str(raw_value).strip()
            elif parse_type == "special_floor":
                # This special parser returns a dictionary
                floor_data = _parse_floor_string(raw_value)
                # We merge this dictionary into our main clean_data
                clean_data.update(floor_data)
            else:
                # Default for unknown types: just cast to string
                clean_data[english_key] = str(raw_value).strip()
        
        except Exception as e:
            print(f"Error parsing key '{english_key}' (raw: '{raw_value}'): {e}", file=sys.stderr)
            clean_data[english_key] = None # Set to null on failure

    # --- Post-processing & Consolidation ---
    
    # 1. Consolidate 'has_elevator'
    # We prioritize the dedicated 'has_elevator' field.
    # If it's None, we check the one we parsed from the floor string.
    dedicated_elevator = clean_data.get('has_elevator')
    if dedicated_elevator is None:
        clean_data['has_elevator'] = clean_data.get('has_elevator_from_string')
        
    # Remove the temporary helper key
    if 'has_elevator_from_string' in clean_data:
        del clean_data['has_elevator_from_string']

    # 2. Clean up 'coop_owns_land' (common boolean-like string)
    if 'coop_owns_land' in clean_data:
        clean_data['coop_owns_land'] = _parse_boolean(clean_data['coop_owns_land'])

    # 3. Parse numeric values from coop strings
    if 'coop_annual_fee_per_sqm' in clean_data:
        clean_data['coop_annual_fee_per_sqm_sek'] = _parse_numeric(clean_data.get('coop_annual_fee_per_sqm'))
        del clean_data['coop_annual_fee_per_sqm'] # remove old string key

    if 'coop_debt_per_sqm' in clean_data:
        clean_data['coop_debt_per_sqm_sek'] = _parse_numeric(clean_data.get('coop_debt_per_sqm'))
        del clean_data['coop_debt_per_sqm'] # remove old string key

    # --- Keep null values for schema consistency ---
    final_data = clean_data
            
    return final_data

# --- HELPER FUNCTIONS END ---


def get_next_data_json(soup: BeautifulSoup) -> Optional[Dict[str, Any]]:
    """
    Finds and parses the __NEXT_DATA__ JSON blob from the HTML.
    This is the most reliable source for most of the data.
    """
    try:
        script_tag = soup.find('script', {'id': '__NEXT_DATA__', 'type': 'application/json'}, string=True)
        if not script_tag:
            print("Error: Could not find __NEXT_DATA__ script tag.", file=sys.stderr)
            return None
        
        json_data = json.loads(script_tag.string)
        return json_data
    except Exception as e:
        print(f"Error parsing __NEXT_DATA__ JSON: {e}", file=sys.stderr)
        return None

def find_listing_data(next_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Navigates the __NEXT_DATA__ JSON to find the main listing data object.
    Checks for multiple known key prefixes.
    """
    try:
        apollo_state = next_data.get("props", {}).get("pageProps", {}).get("__APOLLO_STATE__", {})
        
        for key in apollo_state:
            if key.startswith("ProjectUnit:") or key.startswith("Listing:") or key.startswith("ActivePropertyListing:"):
                return apollo_state[key]
                
        print("Error: Could not find 'ProjectUnit', 'Listing', or 'ActivePropertyListing' data in __APOLLO_STATE__.", file=sys.stderr)
        return None
    except Exception as e:
        print(f"Error navigating JSON to find listing data: {e}", file=sys.stderr)
        return None

def find_value_by_label_text(soup: BeautifulSoup, label: str) -> Optional[str]:
    """
    Finds an element containing the exact label text (e.g., in a <p>) and 
    returns the text of the *next* sibling element.
    Used for Area Statistics and Down Payment.
    """
    try:
        text_node = soup.find(string=re.compile(r'^\s*' + re.escape(label) + r'\s*$', re.IGNORECASE))
        if text_node:
            parent_tag = text_node.parent
            value_tag = parent_tag.find_next_sibling()
            if value_tag:
                return value_tag.get_text(strip=True)
    except Exception:
        pass 
    return None

def find_dt_dd_value(soup: BeautifulSoup, label: str) -> Optional[str]:
    """
    Finds a <dt> tag with the specified label and returns the text of
    the immediately following <dd> tag.
    Used for the main attributes list.
    """
    try:
        dt_tag = soup.find('dt', string=re.compile(r'^\s*' + re.escape(label) + r'\s*$', re.IGNORECASE))
        if dt_tag:
            dd_tag = dt_tag.find_next_sibling('dd')
            if dd_tag:
                # Clean up child tags (like info icons)
                for child in dd_tag.find_all():
                    child.decompose()
                return dd_tag.get_text(strip=True)
    except Exception:
        pass
    return None

def find_brf_info_value(brf_section: Optional[Tag], label: str) -> Optional[str]:
    """
    Finds a label (e.g., "Status") within the <section id='brf-info'>
    and returns its corresponding value (e.g., "Äkta förening").
    This function is safe and handles brf_section being None.
    """
    if not brf_section:
        return None
    
    try:
        p_tag = brf_section.find('p', string=re.compile(r'^\s*' + re.escape(label) + r'\s*$', re.IGNORECASE))
        if p_tag:
            container = p_tag.find_parent(class_=re.compile(r'NestList_nestListItemContent__'))
            if container:
                value_div = container.find('div', class_=re.compile(r'NestList_nestListItemTrailing__'))
                if value_div:
                    value_label = value_div.find('label')
                    if value_label:
                        return value_label.get_text(strip=True)
    except Exception:
        pass
    return None

def get_static_short_description(soup: BeautifulSoup) -> Optional[str]:
    """
    Gets the *static* short description snippet. This block is often
    separate from the main description and is characterized by being
    hidden on mobile.
    """
    try:
        # The key differentiator for this block seems to be the 'hideOnMobile' class.
        desc_div = soup.find('div', class_=re.compile(r'Description_descriptionContainer__.*hideOnMobile__'))
        if desc_div:
            return desc_div.get_text(strip=True)
    except Exception:
        pass
    return None

def get_detailed_description_html(soup: BeautifulSoup) -> Optional[str]:
    """
    Gets the main, long description text from the *HTML* only.
    The description component uses hashed CSS classes that always
    share the prefix 'Description_description__'.
    """
    def has_description_class(tag: Tag) -> bool:
        classes = tag.get('class', [])
        return any(re.search(r'^Description_description__', cls) for cls in classes)

    candidate = None
    for tag in soup.find_all(has_description_class):
        # Prefer the innermost node that actually holds the text,
        # as its parents often include controls like "Visa beskrivning".
        if tag.find(has_description_class):
            continue
        candidate = tag
        break

    if not candidate:
        # Conservative fallback: try the expander text container which
        # wraps the description when the hashed class changes.
        candidate = soup.find('div', class_=re.compile(r'^Expander_textContainer__'))

    if not candidate:
        return None

    text = candidate.get_text(separator="\n", strip=True)
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    for noise in ("Visa mer", "Visa beskrivning", "Dölj beskrivning"):
        text = re.sub(rf'\b{re.escape(noise)}\b', '', text).strip()

    return text or None

def mercator_pixels_to_latlon(x_px: float, y_px: float, zoom: int, tile_size: int = 256) -> Tuple[float, float]:
    world_size = tile_size * (2 ** zoom)
    lon = (x_px / world_size) * 360.0 - 180.0
    lat_rad = math.pi - (2.0 * math.pi * y_px / world_size)
    lat = math.degrees(math.atan(math.sinh(lat_rad)))
    return lat, lon

def extract_button_coordinates(soup: BeautifulSoup) -> Optional[Tuple[float, float]]:
    def matches(tag: Tag) -> bool:
        return (
            isinstance(tag, Tag) and
            tag.get('role') == 'button' and
            tag.get('position')
        )

    candidate = soup.find(matches)
    if not candidate:
        return None
    try:
        lat_str, lon_str = candidate['position'].split(',', 1)
        return float(lat_str.strip()), float(lon_str.strip())
    except (ValueError, KeyError):
        return None

def parse_style_numeric(style: str, prop: str) -> Optional[float]:
    if not style:
        return None
    match = re.search(rf'{prop}\s*:\s*(-?\d+(?:\.\d+)?)px', style)
    return float(match.group(1)) if match else None

def parse_transform_translation(style: str) -> Optional[Tuple[float, float]]:
    if not style:
        return None
    match = re.search(r'transform\s*:\s*matrix\(\s*[^,]+,\s*[^,]+,\s*[^,]+,\s*[^,]+,\s*(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\s*\)', style)
    if match:
        return float(match.group(1)), float(match.group(2))
    return None

def extract_offset_coordinates(soup: BeautifulSoup) -> Optional[Tuple[float, float]]:
    img = soup.find('img', src=re.compile(r'mapsresources-pa\.googleapis\.com/.+!1m5!1m4!1i\d+!2i\d+!3i\d+!4i\d+'))
    if not img:
        return None

    match = re.search(r'!1i(\d+)!2i(\d+)!3i(\d+)!4i(\d+)', img.get('src', ''))
    if not match:
        return None

    zoom = int(match.group(1))
    tile_x = int(match.group(2))
    tile_y = int(match.group(3))
    tile_size = int(match.group(4))

    tile_div = img.parent
    while tile_div and (not isinstance(tile_div, Tag) or 'style' not in tile_div.attrs or 'left' not in tile_div['style']):
        tile_div = tile_div.parent
    if not isinstance(tile_div, Tag):
        return None

    pane_div = tile_div.parent
    while pane_div and (not isinstance(pane_div, Tag) or 'style' not in pane_div.attrs or 'transform' not in pane_div['style']):
        pane_div = pane_div.parent
    if not isinstance(pane_div, Tag):
        return None

    left = parse_style_numeric(tile_div.get('style', ''), 'left')
    top = parse_style_numeric(tile_div.get('style', ''), 'top')
    translation = parse_transform_translation(pane_div.get('style', ''))

    if left is None or top is None or translation is None:
        return None

    trans_x, trans_y = translation
    offset_x = -(left + trans_x)
    offset_y = -(top + trans_y)

    world_x = tile_x * tile_size + offset_x
    world_y = tile_y * tile_size + offset_y

    lat, lon = mercator_pixels_to_latlon(world_x, world_y, zoom, tile_size)
    return lat, lon

def get_json_val(keys_list: list, default: Any = None) -> Any:
    """
    Safely traverses a nested dictionary/list structure.
    Returns default if any key or index is not found or is out of range.
    """
    data = listing_data # Uses the global 'listing_data' from the main function
    if not data:
        return default
    try:
        for key in keys_list:
            if isinstance(data, list) and isinstance(key, int):
                if 0 <= key < len(data):
                    data = data[key]
                else:
                    return default # Index out of range
            elif isinstance(data, dict):
                data = data.get(key)
                if data is None:
                    return default # Key doesn't exist
            else:
                return default # Trying to index into a non-dict/non-list
        return data
    except (KeyError, TypeError, IndexError):
        return default

def extract_property_data(html_path: str) -> Dict[str, Any]:
    """
    Parses the HTML file and extracts property data into a structured dictionary,
    showing values from both the JSON and static HTML where available.
    """
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
    except Exception as e:
        print(f"Error reading file '{html_path}': {e}", file=sys.stderr)
        sys.exit(1)

    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Declare listing_data as global so get_json_val can access it
    global listing_data
    next_data = get_next_data_json(soup)
    listing_data = find_listing_data(next_data) if next_data else {}
    if not listing_data:
        print("Warning: Could not extract data from __NEXT_DATA__. Falling back to HTML scraping only.", file=sys.stderr)

    # --- Gather all values ---

    # Adress & Plats
    gatuadress_json = get_json_val(['streetAddress'])
    gatuadress_h1_tag = soup.find('h1', class_=re.compile(r'Heading_hclHeading__'))
    gatuadress_html = gatuadress_h1_tag.get_text(strip=True) if gatuadress_h1_tag else None
    
    område_json = get_json_val(['area'])
    kommun_json = get_json_val(['municipality', 'fullName'])
    
    område_html = None
    kommun_html = None
    if gatuadress_h1_tag and gatuadress_h1_tag.parent:
        location_div = gatuadress_h1_tag.parent.find_next_sibling('div')
        if location_div:
            location_span = location_div.find('span')
            if location_span:
                location_text = location_span.get_text(strip=True)
                if ',' in location_text:
                    parts = location_text.split(',')
                    område_html = parts[0].strip()
                    kommun_html = parts[1].strip()

    avstand_vatten_html_tag = soup.find('p', string=re.compile(r'till vatten'))
    avstand_hav_html_tag = soup.find('p', string=re.compile(r'till havet'))
    avstand_vatten_html = avstand_vatten_html_tag.get_text(strip=True) if avstand_vatten_html_tag else None
    avstand_hav_html = avstand_hav_html_tag.get_text(strip=True) if avstand_hav_html_tag else None
    vatten_namn_html = _extract_text_in_parentheses(avstand_vatten_html)
    hav_namn_html = _extract_text_in_parentheses(avstand_hav_html)


    button_latlon = extract_button_coordinates(soup)
    offset_latlon = extract_offset_coordinates(soup)
    lat_html_source = {
        "button": button_latlon[0] if button_latlon else None,
        "offset": offset_latlon[0] if offset_latlon else None,
    }
    lon_html_source = {
        "button": button_latlon[1] if button_latlon else None,
        "offset": offset_latlon[1] if offset_latlon else None,
    }

    # Priser & Avgifter
    utgangspris_json = get_json_val(['askingPrice', 'formatted']) or get_json_val(['askingPrice', 'amount'])
    price_span_html_tag = soup.find('span', class_=re.compile(r'ListingPrice_listingPrice__'))
    utgangspris_html = price_span_html_tag.get_text(strip=True) if price_span_html_tag else None
    
    # --- NEW: Find original price and change percentage ---
    original_price_tag = soup.find('div', class_=re.compile(r'ListingPrice_originalPrice__'))
    utgangspris_original_html = original_price_tag.get_text(strip=True) if original_price_tag else None
    
    price_change_tag = soup.find('span', class_=re.compile(r'ListingPrice_priceChange__'))
    utgangspris_forandring_procent_html = price_change_tag.get_text(strip=True) if price_change_tag else None
    # --- END NEW ---

    avgift_json = get_json_val(['fee', 'formatted'])
    avgift_html = find_dt_dd_value(soup, 'Avgift')
    
    driftkostnad_json = get_json_val(['runningCosts', 'formatted'])
    driftkostnad_html = find_dt_dd_value(soup, 'Driftkostnad')
    
    minsta_kontantinsats_html = find_value_by_label_text(soup, 'Minsta kontantinsats')
    
    utgangspris_kvm_json = get_json_val(['squareMeterPrice', 'amount'])
    utgangspris_kvm_html = find_dt_dd_value(soup, 'Pris/m²') # HTML label is 'Pris/m²'

    # Områdesstatistik
    kvm_snitt_html = find_value_by_label_text(soup, 'Kvadratmeterpris (snitt)')
    prisutveckling_html = find_value_by_label_text(soup, 'Prisutveckling')

    # Detaljer & Fastighet
    bostadstyp_json = get_json_val(['housingForm', 'name'])
    bostadstyp_html = find_dt_dd_value(soup, 'Bostadstyp')
    
    upplåtelseform_json = get_json_val(['tenure', 'name'])
    upplåtelseform_html = find_dt_dd_value(soup, 'Upplåtelseform')
    
    antal_rum_json = get_json_val(['numberOfRooms'])
    antal_rum_html = find_dt_dd_value(soup, 'Antal rum')
    
    boarea_json = get_json_val(['livingArea'])
    boarea_html = find_dt_dd_value(soup, 'Boarea')
    
    biarea_json = get_json_val(['supplementalArea'])
    biarea_html = find_dt_dd_value(soup, 'Biarea')
    
    tomtarea_json = get_json_val(['landArea'])
    tomtarea_html = find_dt_dd_value(soup, 'Tomtarea')
    
    byggår_json = get_json_val(['legacyConstructionYear'])
    byggår_html = find_dt_dd_value(soup, 'Byggår')
    
    våning_json = get_json_val(['formattedFloor'])
    våning_html = find_dt_dd_value(soup, 'Våning')
    
    har_balkong_json = "Ja" if any(a.get('kind') == 'BALCONY' and a.get('isAvailable') for a in get_json_val(['relevantAmenities'], [])) else "Nej"
    har_balkong_html = find_dt_dd_value(soup, 'Balkong')
    
    har_uteplats_json = "Ja" if any(a.get('kind') == 'PATIO' and a.get('isAvailable') for a in get_json_val(['relevantAmenities'], [])) else "Nej"
    har_uteplats_html = find_dt_dd_value(soup, 'Uteplats')
    
    har_hiss_json = "Ja" if any(a.get('kind') == 'ELEVATOR' and a.get('isAvailable') for a in get_json_val(['relevantAmenities'], [])) else "Nej"
    har_hiss_html = "Ja" if (våning_html and "hiss finns" in våning_html.lower()) else "Nej"
    
    energiklass_json = get_json_val(['energyClassification', 'classification'])
    energiklass_html = find_dt_dd_value(soup, 'Energiklass')
    
    planerat_tillträde_json = None
    planerat_tillträde_html = find_dt_dd_value(soup, 'Planerat tillträde')

    # Förening
    brf_section = soup.find('section', {'id': 'brf-info'})
    brf_h2_tag = brf_section.find('h2') if brf_section else None
    brf_name_from_h2_html = brf_h2_tag.get_text(strip=True) if brf_h2_tag else None
    forening_namn_json = get_json_val(['housingCooperative', 'name'])
    forening_namn_dtdd_html = find_dt_dd_value(soup, 'Förening')
    # Prioritize the H2 tag, then dt/dd, then JSON
    forening_namn_html_best = brf_name_from_h2_html or forening_namn_dtdd_html

    # Beskrivning
    beskrivning_kort_html = get_static_short_description(soup)
    beskrivning_detaljerad_json = get_json_val(['description'])
    beskrivning_detaljerad_html = get_detailed_description_html(soup)

    # --- Build the final structured JSON ---
    
    schema = {
        "Adress & Plats": {
            "gatuadress": {"json_value": gatuadress_json, "html_value": gatuadress_html},
            "område": {"json_value": område_json, "html_value": område_html},
            "kommun": {"json_value": kommun_json, "html_value": kommun_html},
            "latitud": {"json_value": get_json_val(['coordinate', 'lat']), "html_value": lat_html_source},
            "longitud": {"json_value": get_json_val(['coordinate', 'lon']), "html_value": lon_html_source},
            "avstånd_vatten_m": {"json_value": None, "html_value": avstand_vatten_html},
            "vatten_namn": {"json_value": None, "html_value": vatten_namn_html},
            "avstånd_hav_m": {"json_value": None, "html_value": avstand_hav_html},
            "hav_namn": {"json_value": None, "html_value": hav_namn_html},
        },
        "Priser & Avgifter": {
            "utgångspris_sek": {"json_value": utgangspris_json, "html_value": utgangspris_html},
            "utgångspris_original_sek": {"json_value": None, "html_value": utgangspris_original_html}, # <-- NEW
            "utgångspris_forandring_procent": {"json_value": None, "html_value": utgangspris_forandring_procent_html}, # <-- NEW
            "avgift_månad_sek": {"json_value": avgift_json, "html_value": avgift_html},
            "driftkostnad_ar_sek": {"json_value": driftkostnad_json, "html_value": driftkostnad_html},
            "minsta_kontantinsats_sek": {"json_value": None, "html_value": minsta_kontantinsats_html},
            "utgångspris_per_kvadratmeter_sek": {"json_value": utgangspris_kvm_json, "html_value": utgangspris_kvm_html},
        },
        "Områdesstatistik": {
            "kvadratmeterpris_snitt_sek": {"json_value": None, "html_value": kvm_snitt_html},
            "prisutveckling_procent": {"json_value": None, "html_value": prisutveckling_html},
        },
        "Detaljer & Fastighet": {
            "bostadstyp": {"json_value": bostadstyp_json, "html_value": bostadstyp_html},
            "upplåtelseform": {"json_value": upplåtelseform_json, "html_value": upplåtelseform_html},
            "antal_rum": {"json_value": antal_rum_json, "html_value": antal_rum_html},
            "boarea_kvm": {"json_value": boarea_json, "html_value": boarea_html},
            "biarea_kvm": {"json_value": biarea_json, "html_value": biarea_html},
            "tomtarea_kvm": {"json_value": tomtarea_json, "html_value": tomtarea_html},
            "byggår": {"json_value": byggår_json, "html_value": byggår_html},
            "våning": {"json_value": våning_json, "html_value": våning_html},
            "har_balkong": {"json_value": har_balkong_json, "html_value": har_balkong_html},
            "har_uteplats": {"json_value": har_uteplats_json, "html_value": har_uteplats_html},
            "har_hiss": {"json_value": har_hiss_json, "html_value": har_hiss_html},
            "energiklass": {"json_value": energiklass_json, "html_value": energiklass_html},
            "planerat_tillträde": {"json_value": planerat_tillträde_json, "html_value": planerat_tillträde_html},
            "forening_namn": {"json_value": forening_namn_json, "html_value": forening_namn_html_best},
        },
        "Förening": {
            "ekonomi_status": {"json_value": None, "html_value": find_dt_dd_value(soup, 'Ekonomi')},
            "brf_status": {"json_value": None, "html_value": find_brf_info_value(brf_section, "Status")},
            "brf_ager_marken": {"json_value": None, "html_value": find_brf_info_value(brf_section, "Äger marken")},
            "brf_antal_lagenheter": {"json_value": None, "html_value": find_brf_info_value(brf_section, "Antal lägenheter")},
            "brf_registreringsar": {"json_value": None, "html_value": find_brf_info_value(brf_section, "Registreringsår")},
            "brf_arsavgift": {"json_value": None, "html_value": find_brf_info_value(brf_section, "Årsavgift")},
            "brf_belaning": {"json_value": None, "html_value": find_brf_info_value(brf_section, "Belåning")},
            "brf_arsredovisning": {"json_value": None, "html_value": find_brf_info_value(brf_section, "Årsredovisning")},
        },
        "Beskrivning": {
            "beskrivning_kort": {"json_value": None, "html_value": beskrivning_kort_html},
            "beskrivning_detaljerad": {"json_value": beskrivning_detaljerad_json, "html_value": beskrivning_detaljerad_html},
        }
    }
    
    return schema

if __name__ == "__main__":
    import sys, os, json
    from pathlib import Path
    from typing import Optional, Dict, Any

    # --- Check for --debug flag ---
    run_args = [a for a in sys.argv[1:] if not a.startswith('--')]
    is_debug = '--debug' in sys.argv
    
    # Resolve input HTML path
    if len(run_args) < 1:
        if os.path.exists("dom.html"):
            html_path = "dom.html"
            print("--- Using 'dom.html' from current directory ---", file=sys.stderr)
        else:
            print("Error: Please provide the path to the HTML file as an argument.", file=sys.stderr)
            print("Example: python soup.py /path/to/snapshot/dom.html [--debug]", file=sys.stderr)
            sys.exit(1)
    else:
        html_path = run_args[0]

    html_file = Path(html_path).resolve()
    snapshot_dir = html_file.parent

    # Extract
    listing_data: Optional[Dict[Any, Any]] = {} # Type hint fix
    property_data_raw = extract_property_data(str(html_file))
    property_data_raw = _sanitize_ws(property_data_raw)

    # --- Decide which format to save ---
    if is_debug:
        out_path = snapshot_dir / "parsed_debug.json"
        tmp_path = snapshot_dir / "parsed_debug.json.tmp"
        print(f"--- Writing DEBUG (raw) {out_path.name} ---", file=sys.stderr)
        final_data = property_data_raw
    else:
        out_path = snapshot_dir / "parsed.json"
        tmp_path = snapshot_dir / "parsed.json.tmp"
        print(f"--- Writing CLEAN (flattened) {out_path.name} ---", file=sys.stderr)
        final_data = _clean_and_flatten_data(property_data_raw)


    # Write file
    if final_data:
        try:
            tmp_path.write_text(json.dumps(final_data, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp_path.replace(out_path)
            print(f"Wrote: {out_path}")
            sys.exit(0)
        except Exception as e:
            print(f"Error writing JSON to {out_path}: {e}", file=sys.stderr)
            sys.exit(3)
    else:
        print("No data extracted; nothing written.", file=sys.stderr)
        sys.exit(2)