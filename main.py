import os
import sys
import json
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig
from pydantic import BaseModel, EmailStr
from typing import List, Optional
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from rapidfuzz import process

from xhtml2pdf import pisa

import re
import tempfile
from pathlib import Path
from unicodedata import normalize as u_normalize

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import numpy as np

# ML / fuzzy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
from nltk.stem.snowball import SnowballStemmer

# ----------------------------------------------------
# Basic helpers
# ----------------------------------------------------
def normalize(name: str) -> str:
    return name.lower().split("(")[0].strip()

def normalize_text(s: str) -> str:
    """Lowercase, remove accents, punctuation, collapse whitespace."""
    if not isinstance(s, str):
        return ""
    s = u_normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ----------------------------------------------------
# FastAPI
# ----------------------------------------------------
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# ----------------------------------------------------
# Email config
# ----------------------------------------------------
conf = None
mail_username = os.getenv("MAIL_USERNAME")
mail_password = os.getenv("MAIL_PASSWORD")
mail_from = os.getenv("MAIL_FROM")

# Only create email config if we have valid email credentials (not placeholder values)
if (mail_username and mail_password and mail_from and 
    mail_username != "your_railway_mail_username" and
    mail_password != "your_railway_mail_password" and
    mail_from != "your_railway_mail_from" and
    "@" in mail_from):  # Basic email validation
    conf = ConnectionConfig(
        MAIL_USERNAME=mail_username,
        MAIL_PASSWORD=mail_password,
        MAIL_FROM=mail_from,
        MAIL_PORT=int(os.getenv("MAIL_PORT", 587)),
        MAIL_SERVER=os.getenv("MAIL_SERVER", "smtp.hostinger.com"),
        MAIL_STARTTLS=os.getenv("MAIL_STARTTLS", "True").lower() == "true",
        MAIL_SSL_TLS=os.getenv("MAIL_SSL_TLS", "False").lower() == "true",
        USE_CREDENTIALS=True,
        VALIDATE_CERTS=True
    )
    fastmail = FastMail(conf)

print("DEBUG MAIL CONFIG")
print("MAIL_USERNAME:", os.getenv("MAIL_USERNAME"))
print("MAIL_PASSWORD:", os.getenv("MAIL_PASSWORD"))
print("MAIL_FROM:", os.getenv("MAIL_FROM"))
print("MAIL_PORT:", os.getenv("MAIL_PORT"))
print("MAIL_SERVER:", os.getenv("MAIL_SERVER"))
print("MAIL_STARTTLS:", os.getenv("MAIL_STARTTLS"))
print("MAIL_SSL_TLS:", os.getenv("MAIL_SSL_TLS"))

# ----------------------------------------------------
# Load CSV
# ----------------------------------------------------
csv_path = os.getenv("CSV_FILE_PATH", "cleaned_claimchecker.csv")
print(f"🔍 Looking for CSV file at: {csv_path}")
print(f"📁 Current working directory: {os.getcwd()}")
print(f"📋 Files in current directory: {os.listdir('.')}")

# Check if we're on Railway
if os.getenv("RAILWAY_ENVIRONMENT"):
    print("🚂 Running on Railway environment")
    # Ensure we're looking in the right place
    if not os.path.exists(csv_path):
        # Try alternative paths
        alt_paths = [
            "./cleaned_claimchecker.csv",
            "/app/cleaned_claimchecker.csv",
            "cleaned_claimchecker.csv"
        ]
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                csv_path = alt_path
                print(f"✅ Found CSV at alternative path: {csv_path}")
                break
        else:
            print(f"❌ CSV file not found in any expected location")
            print(f"Available files: {os.listdir('.')}")

if not os.path.exists(csv_path):
    print(f"ERROR: CSV file not found at {csv_path}", file=sys.stderr)
    print(f"Please set the CSV_FILE_PATH environment variable or place your CSV file at {csv_path}", file=sys.stderr)
    sys.exit(1)

try:
    df = pd.read_csv(
        csv_path,
        sep=",",
        engine="python",
        on_bad_lines="warn",
        quotechar='"',
        skip_blank_lines=True,
        header=0
    )
    print(f"✅ Successfully loaded CSV with {len(df)} rows and {len(df.columns)} columns")
    print(f"📊 DataFrame columns: {list(df.columns)}")
    
    # Clean the data
    df = df.applymap(lambda v: v.strip().strip('"') if isinstance(v, str) else v)
    
    print(f"🎯 Sample data - first 3 rows:")
    print(df.head(3).to_string())
    
except Exception as e:
    print(f"❌ Error loading CSV: {e}", file=sys.stderr)
    print(f"Full traceback:", file=sys.stderr)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

text_cols = [
    "Claim Category",
    "Allowed Claims", 
    "Claim",
    "Dosage",
    "Health claim pending European authorisation",
    "Claim Use Notes"
]
for col in text_cols:
    if col not in df.columns:
        df[col] = ""
df[text_cols] = df[text_cols].fillna("").astype(str)

# ----------------------------------------------------
# ✅ LOAD GPT VARIATIONS JSON
# ----------------------------------------------------
import json

GPT_VARIATIONS_PATH = "gpt_claim_variations.json"
VARIATION_LOOKUP = {}

if os.path.exists(GPT_VARIATIONS_PATH):
    with open(GPT_VARIATIONS_PATH, "r") as f:
        gpt_data = json.load(f)
    for entry in gpt_data:
        key = normalize_text(entry.get("Original", ""))
        VARIATION_LOOKUP[key] = entry.get("Variations", [])
    print(f"✅ Loaded {len(VARIATION_LOOKUP)} GPT claim variations")
else:
    print(f"⚠️ GPT variations file not found at {GPT_VARIATIONS_PATH}")

# ----------------------------------------------------
# Load GPT Variations JSON + Helper (FIXED - removed duplicate)
# ----------------------------------------------------
import json
from rapidfuzz import process

try:
    with open("gpt_claim_variations.json", "r", encoding="utf-8") as f:
        GPT_VARIATIONS = json.load(f)

    # Build a lookup dict
    GPT_LOOKUP = {}
    for entry in GPT_VARIATIONS:
        original = entry.get("Original", "").strip()
        if original:
            GPT_LOOKUP[original.lower()] = entry.get("Variations", [])
    
    print(f"✅ Loaded GPT variations lookup with {len(GPT_LOOKUP)} entries")
except Exception as e:
    print(f"⚠️ Error loading GPT variations: {e}")
    GPT_LOOKUP = {}

def get_variations_for_claim(claim: str):
    claim_norm = claim.lower().strip()
    if claim_norm in GPT_LOOKUP:
        variations = GPT_LOOKUP[claim_norm]
        return variations if variations else []
    
    best_match, score, _ = process.extractOne(claim_norm, GPT_LOOKUP.keys())
    if score > 80 and best_match:
        variations = GPT_LOOKUP[best_match]
        return variations if variations else []
    
    return []


# ----------------------------------------------------
# CATEGORY LEXICON
# ----------------------------------------------------
CATEGORY_LEXICON = {
    "immune":        ["immune", "immunity", "defence", "defense", "resistance"],
    "digestive":     ["digest", "stomach", "gut", "intestinal", "flatulence", "bowel", "microbiome", "microbiota"],
    "respiratory":   ["respiratory", "bronchial", "airway", "throat", "cough", "lungs", "bronchi"],
    "sleep_relax":   ["sleep", "relax", "stress", "calm", "anxiety", "restful", "nervous tension"],
    "joint_bone":    ["joint", "bone", "arthritis", "muscle", "tendon", "cartilage", "mobility", "flexibility"],
    "cardio":        ["cardio", "heart", "vascular", "circulation", "blood pressure", "capillary", "blood vessel", "artery", "vein"],
    "liver_detox":   ["liver", "hepatic", "detox", "detoxification", "bile"],
    "skin_hair_nails":["skin", "hair", "nails", "derma", "dermatological"],
    "cognitive":     ["cognitive", "memory", "concentration", "alertness", "focus", "mental performance"],
    "energy_fatigue":["energy", "energetic", "vitality", "fatigue", "tiredness", "stamina", "strengthen", "physical capacities", "physical fatigue", "tonic"],
    "antioxidant":   ["antioxidant", "oxidative", "free radical", "oxidation"],
    "urinary":       ["urinary", "renal", "kidney", "urination"],
    "reproductive":  ["reproductive", "libido", "fertility", "sperm", "male vitality", "sexual desire"],
    "menopause":     ["menopause", "menopausal", "hormone", "hormonal", "estrogen", "phytoestrogen"],
    "general_health":["general", "health", "wellbeing", "wellness", "vitality"],
    "adaptogen":     ["adaptogen", "adaptogenic", "resistance", "physiological resistance", "organism", "ambiance conditions", "severe ambiance", "defensive mechanism"],
    "lactation":     ["lactation", "breastfeeding", "nursing", "milk"],
    "menstrual":     ["menstrual", "menstruation", "period", "cycle", "flow"],
    "metabolic":     ["metabolic", "weight", "obesity", "metabolism"],
    "blood_sugar":   ["glycaemic", "glycemic", "blood sugar", "glucose", "insulin"],
    "cholesterol":   ["cholesterol", "lipid", "triglyceride", "hdl", "ldl"],
    "prebiotic":     ["prebiotic", "prebiotics"],
    "inflammation":  ["inflammation", "inflammatory", "anti-inflammatory", "inflamed"]
}

# ----------------------------------------------------
# Claim splitting & categorisation
# ----------------------------------------------------
SPLIT_RE = re.compile(r"(?:\\n|\n|;|•|\u2022|^-\s+|\s-\s+)+", flags=re.MULTILINE)
LABEL_RE = re.compile(r"^\s*(on[-\s]?hold:?|on[-\s]?hold\s*\\?:?)\s*", re.IGNORECASE)
TOKEN_RE = re.compile(r"[a-z0-9]+")

stemmer = SnowballStemmer("english")

def clean_claim(c: str) -> str:
    c = LABEL_RE.sub("", c.strip())
    c = re.sub(r"^\d+[\)\.:-]\s*", "", c)
    c = re.sub(r"\s+", " ", c).strip()
    return c

def split_claims(raw) -> list[str]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    pieces = SPLIT_RE.split(str(raw))
    out = []
    for p in pieces:
        p = clean_claim(p)
        if len(p) >= 3:
            out.append(p)
    # dedupe (preserve order)
    seen = set(); final = []
    for p in out:
        p_norm = p.lower()
        if p_norm not in seen:
            seen.add(p_norm)
            final.append(p)
    return final

def tokenize(s: str) -> list[str]:
    return TOKEN_RE.findall(s.lower())

def stems_of(words: list[str]) -> set[str]:
    return {stemmer.stem(w) for w in words}

def claim_stems(text: str) -> set[str]:
    return stems_of(tokenize(text))

# Build category stems
CATEGORY_STEMS = {
    cat: stems_of([tok for phrase in words for tok in tokenize(phrase)])
    for cat, words in CATEGORY_LEXICON.items()
}

def assign_best_category_from_stems(stems: set[str]) -> str:
    best_cat, best_score = "uncategorized", 0
    for cat, cat_stems in CATEGORY_STEMS.items():
        score = len(stems & cat_stems)
        if score > best_score:
            best_cat, best_score = cat, score
    return best_cat

def category_for_query(query: str) -> str:
    return assign_best_category_from_stems(claim_stems(normalize_text(query)))

# ----------------------------------------------------
# Build per-claim table & TF-IDF index
# ----------------------------------------------------
def build_claim_index(df: pd.DataFrame):
    rows = []
    for i, r in df.iterrows():
        ingredient = r.get("Ingredient", "")
        country = r.get("Country", "")
        allowed = r.get("Allowed Claims", "") or r.get("Claim", "")
        dosage  = r.get("Dosage", "")
        # Use the actual categories from the CSV instead of guessing
        categories = r.get("Categories", "")

        for c in split_claims(allowed):
            c_norm = normalize_text(c)
            # Use the categories from CSV, fallback to keyword matching if empty
            if categories and categories.strip():
                # Split composite categories and create separate rows for each
                category_list = [cat.strip().lower() for cat in categories.split(",")]
                for cat in category_list:
                    rows.append({
                        "Ingredient": ingredient,
                        "Country": country,
                        "claim": c,
                        "claim_norm": c_norm,
                        "category": cat,
                        "Dosage": dosage,   # ✅ dosage meenemen
                        "row_idx": i
                    })
            else:
                cat = assign_best_category_from_stems(claim_stems(c_norm))
                rows.append({
                    "Ingredient": ingredient,
                    "Country": country,
                    "claim": c,
                    "claim_norm": c_norm,
                    "category": cat,
                    "Dosage": dosage,   # ✅ dosage meenemen
                    "row_idx": i
                })

    df_claims = pd.DataFrame(rows)

    if df_claims.empty:
        return df_claims, None, None

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.99
    )
    tfidf_matrix = vectorizer.fit_transform(df_claims["claim_norm"])
    return df_claims, vectorizer, tfidf_matrix

df_claims, vectorizer, tfidf_matrix = build_claim_index(df)

# ----------------------------------------------------
# SVG icons + render helpers
# ----------------------------------------------------
icon_claim_category = '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 28 28" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-boxes-icon lucide-boxes w-6 h-6 text-indigo-500"><path d="M2.97 12.92A2 2 0 0 0 2 14.63v3.24a2 2 0 0 0 .97 1.71l3 1.8a2 2 0 0 0 2.06 0L12 19v-5.5l-5-3-4.03 2.42Z"/><path d="m7 16.5-4.74-2.85"/><path d="m7 16.5 5-3"/><path d="M7 16.5v5.17"/><path d="M12 13.5V19l3.97 2.38a2 2 0 0 0 2.06 0l3-1.8a2 2 0 0 0 .97-1.71v-3.24a2 2 0 0 0-.97-1.71L17 10.5l-5 3Z"/><path d="m17 16.5-5-3"/><path d="m17 16.5 4.74-2.85"/><path d="M17 16.5v5.17"/><path d="M7.97 4.42A2 2 0 0 0 7 6.13v4.37l5 3 5-3V6.13a2 2 0 0 0-.97-1.71l-3-1.8a2 2 0 0 0-2.06 0l-3 1.8Z"/><path d="M12 8 7.26 5.15"/><path d="m12 8 4.74-2.85"/><path d="M12 13.5V8"/></svg>'''
icon_allowed_claims = '''<svg xmlns="http://www.w3.org/2000/svg" 2ie5h="24" height="24" viewBox="0 0 28 28" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-database-zap w-6 h-6 text-indigo-500"><ellipse cx="12" cy="5" rx="9" ry="3" /><path d="M3 5V19A9 3 0 0 0 15 21.84" /><path d="M21 5V8" /><path d="M21 12L18 17H22L19 22" /><path d="M3 12A9 3 0 0 0 14.59 14.87" /></svg>'''
icon_dosage = '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 28 28" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-test-tube-diagonal w-6 h-6 text-indigo-500"><path d="M21 7 6.82 21.18a2.83 2.83 0 0 1-3.99-.01a2.83 2.83 0 0 1 0-4L17 3"/><path d="m16 2 6 6"/><path d="M12 16H4"/></svg>'''
icon_pending = '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 28 28" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-notebook-tabs w-6 h-6 text-indigo-500"><path d="M2 6h4"/><path d="M2 10h4"/><path d="M2 14h4"/><path d="M2 18h4"/><rect width="16" height="20" x="4" y="2" rx="2"/><path d="M15 2v20"/><path d="M15 7h5"/><path d="M15 12h5"/><path d="M15 17h5"/></svg>'''
icon_notes = '''<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 28 28" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-octagon-alert w-6 h-6 text-indigo-500"><path d="M12 16h.01"/><path d="M12 8v4"/><path d="M15.312 2a2 2 0 0 1 1.414.586l4.688 4.688A2 2 0 0 1 22 8.688v6.624a2 2 0 0 1-.586 1.414l-4.688 4.688a2 2 0 0 1-1.414.586H8.688a2 2 0 0 1-1.414-.586l-4.688-4.688A2 2 0 0 1 2 15.312V8.688a2 2 0 0 1 .586-1.414l4.688-4.688A2 2 0 0 1 8.688 2z"/></svg>'''

def section(title, content, icon_svg, ingredient=None):
    content = str(content).replace("\\n", "\n")
    lines = [
        f'<li data-claim="{line.strip()}">{line.strip()}<div class="claim-variations mt-1 ml-4"></div></li>'
        for line in content.replace(";", "\n").split("\n")
        if line.strip()
    ]

    items_html = (
        f"<ul class='list-disc pl-6 space-y-1'>{''.join(lines)}</ul>"
        if len(lines) > 1 else f"<p>{content}</p>"
    )
    
    # Create copy icon
    copy_icon = f"""
    <button onclick="copyContainerContent(this)" class="copy-icon" title="Copy all content">
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
        </svg>
    </button>
    """
    
    return f"""
    <details class="printable-section group bg-white p-4 rounded-xl shadow transition-all relative">
      <summary class="flex items-center justify-between cursor-pointer list-none">
        <div class="flex items-center gap-2 text-gray-800 font-semibold">
          {icon_svg}
          <span>{title}</span>
        </div>
        <svg class="w-4 h-4 text-gray-500 group-open:rotate-180 transition-transform"
             xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"
             stroke="currentColor">
             <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                   d="M19 9l-7 7-7-7" />
        </svg>
      </summary>
      <div class="mt-4 text-gray-700 text-sm leading-relaxed">
        {items_html}
      </div>
      {copy_icon}
    </details>
    """
from html import escape
def render_claim_card_collapsible(title, claims, dosage, idx, add_rewrite=True, icon_html=""):
    allowed_claims_class = " allowed-claims-section" if title.strip().lower() == "allowed claims" else ""
    open_attr = " open" if idx == 1 else ""

    # Build dosage block if provided
    dosage_block = ""
    if dosage and dosage.strip():
        lines = [ln.strip() for ln in re.split(r"(?:\\n|\n|;)+", dosage) if ln.strip()]
        dosage_items = "".join(f"<li>{escape(ln)}</li>" for ln in lines)
        dosage_block = f"""
        <details class="mt-3 bg-gray-50 rounded-md p-3">
            <summary class="cursor-pointer list-none flex items-center gap-2 text-sm font-medium text-gray-700">
                Dosage
            </summary>
            <ul class="mt-2 list-disc pl-6 space-y-1 text-sm text-gray-700">
                {dosage_items}
            </ul>
        </details>
        """

    # Build the claims HTML as a list
    claims_html = ""
    for c in claims:
        claims_html += f"""
        <li data-claim="{escape(c)}" data-section="allowed" class="mb-2 block">
            <div class="flex items-start gap-2 flex-wrap w-full">
                <span class="flex-1">{escape(c).capitalize()}</span>
                <button type="button"
                        class="view-variations-btn flex items-center gap-1 px-3 py-1 rounded-full bg-indigo-100 text-[#4F46E5] font-medium text-xs hover:bg-indigo-200 transition"
                        data-claim="{escape(c)}" title="View Variations">
                    <svg xmlns="http://www.w3.org/2000/svg" width="6" height="6" fill="none" stroke="#4F46E5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 14a1 1 0 0 1-.78-1.63l9.9-10.2a.5.5 0 0 1 .86.46l-1.92 6.02A1 1 0 0 0 13 10h7a1 1 0 0 1 .78 1.63l-9.9 10.2a.5.5 0 0 1-.86-.46l1.92-6.02A1 1 0 0 0 11 14z"/></svg>
                    View Variations
                </button>
                <div class="claim-variations ml-2"></div>
            </div>
        </li>
        """

    # Create copy icon (only for non-Allowed Claims sections to avoid duplicates)
    copy_icon = ""
    if title.strip().lower() != "allowed claims":
        copy_icon = f"""
        <button onclick="copyContainerContent(this)" class="copy-icon" title="Copy all content">
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
            </svg>
        </button>
        """

    # Now return the HTML block
    return f"""
    <details{open_attr} class="bg-white rounded-xl shadow p-4 mb-3 group printable-section{allowed_claims_class} relative">
        <summary class="cursor-pointer list-none flex items-center justify-between">
            <span class="font-semibold text-gray-800 flex items-center gap-2">{icon_html} {escape(title)}</span>
            <svg class="w-4 h-4 text-gray-500 group-open:rotate-180 transition-transform"
                xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"
                stroke="currentColor">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7" />
            </svg>
        </summary>
        <ul class="list-disc pl-6 mt-3 space-y-1 text-sm text-gray-700">
            {claims_html}
        </ul>
        {dosage_block}
        {copy_icon}
    </details>
    """


# ----------------------------------------------------
# ROUTES
# ----------------------------------------------------
@app.get("/_columns", response_class=JSONResponse)
async def list_columns():
    return {"columns": df.columns.tolist()}

@app.get("/categories", response_class=JSONResponse)
def list_categories():
    return {"categories": list(CATEGORY_LEXICON.keys())}

@app.get("/lexicon", response_class=JSONResponse)
def get_lexicon():
    # Create a lexicon based on actual CSV categories and their keywords
    csv_lexicon = {}
    
    try:
        # Get unique categories from CSV
        if "Categories" in df.columns:
            categories = df["Categories"].dropna().unique()
            
            # Extract main categories (not composite ones)
            main_categories = set()
            for category in categories:
                if category and category.strip():
                    # Split composite categories and add individual ones
                    category_parts = [cat.strip().lower() for cat in category.split(",")]
                    main_categories.update(category_parts)
            
            # Now process each main category
            for main_category in main_categories:
                if main_category:
                    # Find all rows that contain this main category
                    matching_rows = df[df["Categories"].str.contains(main_category, case=False, na=False)]
                    keywords = []
                    
                    # Check if Category_Keywords column exists
                    category_keywords_col = None
                    for col in df.columns:
                        if "Category_Keywords" in col:
                            category_keywords_col = col
                            break
                    
                    if category_keywords_col:
                        for _, row in matching_rows.iterrows():
                            category_keywords = row.get(category_keywords_col, "")
                            # Only process if it's a string and not empty
                            if isinstance(category_keywords, str) and category_keywords.strip():
                                # Extract keywords from the Category_Keywords field
                                # Format is "category: keyword1, keyword2, keyword3"
                                if ":" in category_keywords:
                                    keywords_part = category_keywords.split(":", 1)[1]
                                    if keywords_part:
                                        # Split by comma and clean up
                                        keywords.extend([kw.strip() for kw in keywords_part.split(",")])
                                else:
                                    keywords.extend([kw.strip() for kw in category_keywords.split(",")])
                    
                    # Remove duplicates and empty strings
                    unique_keywords = list(set([kw for kw in keywords if kw]))
                    csv_lexicon[main_category] = unique_keywords
    
        # Fallback to original lexicon for categories not in CSV
        for cat, keywords in CATEGORY_LEXICON.items():
            if cat not in csv_lexicon:
                csv_lexicon[cat] = keywords
    
    except Exception as e:
        print(f"Error in lexicon endpoint: {e}")
        # Return original lexicon as fallback
        return CATEGORY_LEXICON
    
    return csv_lexicon

@app.get("/_cat", response_class=JSONResponse)
def debug_category(q: str):
    return {"query": q, "category": category_for_query(q)}

@app.get("/health")
async def health_check():
    """Health check endpoint for Railway"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "data_loaded": len(df) if 'df' in globals() else 0,
        "variations_loaded": len(GPT_LOOKUP) if 'GPT_LOOKUP' in globals() else 0
    }

@app.get("/", response_class=HTMLResponse)
def read_form(request: Request):
    ingredients = sorted(df["Ingredient"].dropna().unique())
    countries = sorted(df["Country"].dropna().unique())
    return templates.TemplateResponse("index.html", {
        "request": request,
        "ingredients": ingredients,
        "countries": countries
    })

# ---------- Ingredient -> Claims ----------
@app.post("/search-by-ingredient", response_class=HTMLResponse)
async def search_by_ingredient(ingredient: str = Form(...), country: str = Form(...)):
    try:
        print(f"🔍 Searching for ingredient: '{ingredient}' in country: '{country}'")
        print(f"📊 DataFrame shape: {df.shape}")
        print(f"📋 DataFrame columns: {list(df.columns)}")
        
        # Check if 'Ingredient' column exists
        if 'Ingredient' not in df.columns:
            print(f"❌ 'Ingredient' column not found! Available columns: {list(df.columns)}")
            return HTMLResponse(
                "<p class='text-gray-600'>Error: Ingredient column not found in data.</p>",
                status_code=200
            )
        
        # Get all rows for this ingredient across all countries
        matches = df[df["Ingredient"].str.lower() == ingredient.lower()]
        print(f"🎯 Found {len(matches)} matches for ingredient '{ingredient}'")
        
        if matches.empty:
            # Let's see what ingredients are available
            available_ingredients = df["Ingredient"].unique()
            print(f"📝 Available ingredients (first 10): {available_ingredients[:10]}")
            return HTMLResponse(
                "<p class='text-gray-600'>No claims found for this ingredient.</p>",
                status_code=200
            )

        # Collect all unique claims across all countries
        all_claims = set()
        all_dosages = set()
        all_categories = set()
        all_pending = set()
        all_notes = set()
        
        for _, row in matches.iterrows():
            # Get the main claim from the "Claim" column
            claim = row.get("Claim", "")
            if claim and claim.strip():
                all_claims.add(claim.strip())
            
            dosage = row.get("Dosage", "")
            if dosage and dosage.strip():
                all_dosages.add(dosage.strip())
                
            category = row.get("Categories", "")
            if category and category.strip():
                all_categories.add(category.strip())
                
            pending = row.get("Health claim pending European authorisation", "")
            if pending and pending.strip():
                all_pending.add(pending.strip())
                
            notes = row.get("Claim Use Notes", "")
            if notes and notes.strip():
                all_notes.add(notes.strip())

        print(f"📈 Collected {len(all_claims)} unique claims, {len(all_dosages)} dosages")

        if not all_claims:
            return HTMLResponse(
                "<p class='text-gray-600'>No claims found for this ingredient.</p>",
                status_code=200
            )

        # Convert to sorted lists for consistent ordering
        unique_claims = sorted(list(all_claims))
        unique_dosages = sorted(list(all_dosages))
        unique_categories = sorted(list(all_categories))
        unique_pending = sorted(list(all_pending))
        unique_notes = sorted(list(all_notes))
        
        # Combine all dosages into one string
        combined_dosage = "\n".join(unique_dosages) if unique_dosages else ""
        combined_categories = "\n".join(unique_categories) if unique_categories else ""
        combined_pending = "\n".join(unique_pending) if unique_pending else ""
        combined_notes = "\n".join(unique_notes) if unique_notes else ""

        # Format categories like the image but without the number
        formatted_categories_html = "N/A"
        if unique_categories:
            # Create styled category tags like the image
            category_tags = []
            for cat in unique_categories:
                category_tags.append(f'<span class="inline-block bg-indigo-100 text-indigo-800 text-sm font-medium px-3 py-1 rounded-full mr-2 mb-2">{cat}</span>')
            formatted_categories_html = "".join(category_tags)

        # Create the multi-section layout like check-claims
        parts = [
            f"<h2 class='text-2xl font-bold text-gray-800 mb-6'>{ingredient} — All Countries</h2>",
            section("Claim Category", formatted_categories_html, icon_claim_category),
            render_claim_card_collapsible(
                "Allowed Claims",
                unique_claims,
                "",  # Remove dosage from Allowed Claims container
                1,
                add_rewrite=True,
                icon_html=icon_allowed_claims
            ),
            section("Dosage", combined_dosage, icon_dosage),
            section("Health Claim Pending European Authorisation", combined_pending, icon_pending),
            section("Claim Use Notes", combined_notes, icon_notes),
        ]

        html_content = "<div class='space-y-6'>" + "".join(parts) + "</div>"
        return HTMLResponse(html_content, status_code=200)

    except Exception as e:
        import traceback
        print(f"❌ Error in search_by_ingredient: {e}")
        traceback.print_exc()
        return HTMLResponse(
            f"<p style='color: red;'>Server error: {str(e)}</p>",
            status_code=500
        )

# ---------- Claim -> Ingredients ----------
@app.post("/search-by-claim", response_class=HTMLResponse)
async def search_by_claim(
    claim: str = Form(""),
    country: str = Form(...),
    category: Optional[str] = Form(None)
):
    """
    Claim -> Ingredients:
    - Als category gegeven is en claim leeg is: toon alle ingrediënten in die categorie (geen ranking).
    - Als claim is ingevuld (met of zonder category): gebruik TF-IDF + RapidFuzz ranking binnen de (gekozen of afgeleide) categorie.
    """
    try:
        if df_claims.empty or tfidf_matrix is None:
            return HTMLResponse("<p class='text-gray-600'>No claim data indexed.</p>", status_code=200)

        query_norm = normalize_text(claim or "")

        # 1) kies category
        if category:
            query_category = category
        else:
            # als user geen category koos, leiden we hem af op basis van het (eventuele) keyword
            query_category = category_for_query(query_norm)

        # 2) filter by country + category
        mask = (
            (df_claims["Country"].str.lower() == country.lower()) &
            (df_claims["category"] == query_category)
        )
        sub = df_claims[mask].reset_index(drop=True)

        if sub.empty:
            return HTMLResponse(
                f"<p class='text-gray-600'>No matching ingredients found for this claim in {country} (matched category: <b>{query_category}</b>).</p>",
                status_code=200
            )

        # --- PAD 1: GEEN claim/keyword ingevuld → géén ranking, gewoon tonen ---
        if not (claim.strip()):
            TOP_PER_ING = 3
            cards = []
            for idx, (ing, g) in enumerate(sub.groupby("Ingredient", sort=False), start=1):
                seen = set()
                cleaned = []
                for c in g["claim"]:
                    c_norm_l = c.lower().strip()
                    if c_norm_l in seen:
                        continue
                    seen.add(c_norm_l)
                    cleaned.append(c)
                    if len(cleaned) >= TOP_PER_ING:
                        break

                dosage_vals = [d for d in g.get("Dosage", pd.Series([], dtype=str)).astype(str) if d and d.strip()]
                dosage = dosage_vals[0] if dosage_vals else ""

                if cleaned:
                    cards.append(render_claim_card_collapsible(ing, cleaned, dosage, idx, add_rewrite=True))

            header = f"""
              <div class="flex items-center gap-2 mb-4">
                <span class="inline-flex items-center rounded-full bg-indigo-100 text-indigo-700 text-xs font-medium px-2 py-1">
                  Category: {query_category}
                </span>
                <span class="text-xs text-gray-400">({len(cards)} ingredient(s))</span>
              </div>
            """
            return HTMLResponse(header + "".join(cards), status_code=200)

        # --- PAD 2: WÉL claim/keyword ingevuld → ranking met TF-IDF + RapidFuzz ---
        sub_matrix = tfidf_matrix[mask.values]

        q_vec = vectorizer.transform([query_norm])
        cos_scores = cosine_similarity(q_vec, sub_matrix).ravel()
        fuzz_scores = np.array([fuzz.token_set_ratio(query_norm, t) / 100.0 for t in sub["claim_norm"]])

        alpha = 0.7
        blended = alpha * cos_scores + (1 - alpha) * fuzz_scores

        sub = sub.assign(_cos=cos_scores, _fuzz=fuzz_scores, _score=blended)
        sub = sub[sub["_score"] > 0.05].sort_values("_score", ascending=False).head(200)

        if sub.empty:
            return HTMLResponse(
                "<p class='text-gray-600'>No matching ingredients found after scoring.</p>",
                status_code=200
            )

        TOP_PER_ING = 3
        cards = []
        for idx, (ing, g) in enumerate(sub.groupby("Ingredient", sort=False), start=1):
            seen = set()
            cleaned = []
            for c in g["claim"]:
                c_norm_l = c.lower().strip()
                if c_norm_l in seen:
                    continue
                seen.add(c_norm_l)
                cleaned.append(c)
                if len(cleaned) >= TOP_PER_ING:
                    break

            dosage_vals = [d for d in g.get("Dosage", pd.Series([], dtype=str)).astype(str) if d and d.strip()]
            dosage = dosage_vals[0] if dosage_vals else ""

            if cleaned:
                cards.append(render_claim_card_collapsible(ing, cleaned, dosage, idx, add_rewrite=True))

        if not cards:
            return HTMLResponse(
                f"<p class='text-gray-600'>No matching ingredients found in {country} (category: {query_category}).</p>",
                status_code=200
            )

        header = f"""
          <div class="flex items-center gap-2 mb-4">
            <span class="inline-flex items-center rounded-full bg-indigo-100 text-indigo-700 text-xs font-medium px-2 py-1">
              Matched category: {query_category}
            </span>
            <span class="text-xs text-gray-400">({len(cards)} ingredient(s))</span>
          </div>
        """
        return HTMLResponse(header + "".join(cards), status_code=200)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return HTMLResponse(
            f"<p class='text-red-600'>Server error: {str(e)}</p>",
            status_code=500
        )

# ---------- Get GPT Variations for a Single Claim ----------
@app.get("/get-variations", response_class=JSONResponse)
async def get_gpt_variations(claim: str):
    """
    Returns GPT-generated variations for a given claim.
    - Matches exact claim first.
    - Falls back to fuzzy match if needed.
    """
    variations = get_variations_for_claim(claim)
    if not variations:
        return {"claim": claim, "variations": [], "status": "no_match"}
    return {"claim": claim, "variations": variations, "status": "ok"}


# ---------- Check claims (detailed) ----------
@app.post("/check-claims")
async def check_claims(ingredient: str = Form(...), country: str = Form(...)):
    try:
        subset = df[
            (df["Ingredient"].str.lower() == ingredient.strip().lower()) &
            (df["Country"].str.lower() == country.strip().lower())
        ]
        if subset.empty:
            return HTMLResponse(
                "<p class='text-red-600'>No matching ingredient(s) found for that country.</p>",
                status_code=200
            )

        row = subset.iloc[0]

        claims_text = row.get("Allowed Claims", "") or row.get("Claim", "")
        parts = [
            f"<h2 class='text-2xl font-bold text-gray-800 mb-6'>{row['Ingredient']} — {row['Country']}</h2>",
            section("Claim Category", row["Claim Category"], icon_claim_category),
            render_claim_card_collapsible(
    "Allowed Claims",
    split_claims(claims_text),
    "",
    1,
    add_rewrite=True,
    icon_html=icon_allowed_claims   # ✅ this adds the icon
),


            section("Dosage", row["Dosage"], icon_dosage),
            section("Health Claim Pending European Authorisation", row["Health claim pending European authorisation"], icon_pending),
            section("Claim Use Notes", row["Claim Use Notes"], icon_notes),
        ]

        html_content = "<div class='space-y-6'>" + "".join(parts) + "</div>"
        return HTMLResponse(html_content, status_code=200)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# ---------- Email (PDF) ----------
class EmailRequest(BaseModel):
    emails: List[EmailStr]
    html: str

@app.post("/send-email")
async def send_email(email_request: EmailRequest):
    try:
        if conf is None:
            raise HTTPException(status_code=500, detail="Email configuration not available")
        
        # Create enhanced PDF with logo, date/time, and better formatting
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create the enhanced HTML for PDF
        enhanced_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
                .header {{ text-align: center; margin-bottom: 30px; border-bottom: 2px solid #6366f1; padding-bottom: 20px; }}
                .logo-container {{ display: flex; align-items: center; justify-content: center; margin-bottom: 15px; }}
                .logo-svg {{ width: 60px; height: 60px; margin-right: 15px; }}
                .logo-text {{ font-size: 24px; font-weight: bold; color: #213B4E; }}
                .subtitle {{ color: #5F66D3; font-size: 14px; margin-bottom: 10px; }}
                .datetime {{ color: #666; font-size: 12px; }}
                .content {{ margin-top: 20px; }}
                .ingredient-section {{ margin-bottom: 30px; page-break-inside: avoid; }}
                .ingredient-title {{ font-size: 18px; font-weight: bold; color: #213B4E; margin-bottom: 15px; border-left: 4px solid #5F66D3; padding-left: 10px; }}
                .section-title {{ font-size: 14px; font-weight: bold; color: #5F66D3; margin: 15px 0 8px 0; }}
                .claim-item {{ margin: 8px 0; padding-left: 15px; }}
                .claim-bullet {{ color: #6366f1; }}
                .dosage {{ background-color: #f3f4f6; padding: 8px; border-radius: 4px; margin: 10px 0; }}
                .notes {{ font-style: italic; color: #666; margin: 10px 0; }}
                .category-tag {{ display: inline-block; background-color: #e0e7ff; color: #3730a3; padding: 4px 8px; border-radius: 12px; font-size: 12px; margin: 2px; }}
                .footer {{ margin-top: 30px; text-align: center; color: #666; font-size: 12px; border-top: 1px solid #e5e7eb; padding-top: 20px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <div class="logo-container">
                    <svg class="logo-svg" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 5000 5000" preserveAspectRatio="xMidYMid meet">
                        <g class="bar-group g1">
                            <path class="bar bar1" fill="#213B4E" d="M1549 2926 c-135 -133 -251 -251 -257 -263 -9 -16 -12 -232 -12 -806
                              0 -860 -3 -825 57 -794 24 13 380 343 441 409 l22 24 -2 836 -3 835 -246 -241z"/>
                        </g>
                        <g class="bar-group g2">
                            <path class="bar bar2" fill="#5F66D3" d="M2540 3395 c-80 -77 -183 -176 -229 -220 -46 -44 -96 -98 -112 -119
                              l-29 -39 0 -964 c0 -882 1 -964 16 -970 9 -3 31 1 48 11 31 17 420 373 444
                              407 9 14 12 230 10 1026 l-3 1008 -145 -140z"/>
                        </g>
                        <g class="bar-group g3">
                            <path class="bar bar3" fill="#5F66D3" d="M3503 3991 c-37 -32 -148 -139 -245 -238 l-178 -179 0 -1028 0 -1027
                              28 21 c74 58 451 433 466 464 14 31 16 128 16 1034 0 618 -4 1003 -9 1007 -6
                              3 -41 -21 -78 -54z"/>
                        </g>
                    </svg>
                    <div class="logo-text">ClaimSafer™</div>
                </div>
                <div class="subtitle">Ingredient-based EU compliance insights</div>
                <div class="datetime">Generated on: {current_time}</div>
            </div>
            
            <div class="content">
                {email_request.html}
            </div>
            
            <div class="footer">
                <p>This report was generated by ClaimSafer™ Ingredient Vault™</p>
                <p>For questions, contact your compliance team</p>
            </div>
        </body>
        </html>
        """
            
        pdf_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        pisa_status = pisa.CreatePDF(enhanced_html, dest=pdf_file)
        pdf_file.close()

        if pisa_status.err:
            raise HTTPException(status_code=500, detail="PDF generatie mislukt")

        message = MessageSchema(
            subject="Your ClaimSafer™ Report",
            recipients=email_request.emails,
            body="Bijgevoegd vind je je ClaimSafer™ PDF rapport.",
            subtype="plain",
            attachments=[pdf_file.name]
        )

        fm = FastMail(conf)
        await fm.send_message(message)
        Path(pdf_file.name).unlink(missing_ok=True)

        return {"message": "Email + PDF verstuurd"}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Email sending failed: {e}")


# ✅ ---------- Rewrite Claim ----------
from fastapi import Body

@app.post("/rewrite-claim", response_class=JSONResponse)
async def rewrite_claim(data: dict = Body(...)):
    claim = data.get("claim", "")
    if not claim.strip():
        return {"success": False, "message": "No claim provided"}

    # ✅ Dummy rewrite for now
    rewritten = f"✅ Rewritten: {claim}"

    return {"success": True, "rewritten": rewritten}
