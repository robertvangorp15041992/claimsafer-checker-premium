import uuid
from datetime import datetime, timedelta
from collections import defaultdict
import os
import sys
import json
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, Request, Form, HTTPException, Depends, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
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

# Authentication imports
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
import stripe

# Load environment variables
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
import time

# Database imports
from sqlalchemy import create_engine, Column, String, Integer, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Stripe configuration
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./premium_users.db")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security
security = HTTPBearer()

# Create FastAPI app
app = FastAPI(title="ClaimSafer Premium", version="1.0.0")

# Database Models
class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    subscription_tier = Column(String, default="basic")  # basic, premium
    subscription_status = Column(String, default="active")  # active, cancelled, expired
    stripe_customer_id = Column(String, nullable=True)
    stripe_subscription_id = Column(String, nullable=True)
    searches_used = Column(Integer, default=0)
    searches_limit = Column(Integer, default=50)  # 50 for basic, unlimited for premium
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

# Create tables
Base.metadata.create_all(bind=engine)

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Authentication functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            return None
        return email
    except JWTError:
        return None

# Dependency to get current user
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    token = credentials.credentials
    email = verify_token(token)
    if email is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user

# Pydantic models
class UserCreate(BaseModel):
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

# Authentication endpoints
@app.post("/api/register", response_model=Token)
async def register(user: UserCreate, db: Session = Depends(get_db)):
    # Check if user already exists
    existing_user = db.query(User).filter(User.email == user.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create new user
    hashed_password = get_password_hash(user.password)
    db_user = User(
        email=user.email,
        hashed_password=hashed_password,
        subscription_tier="basic",
        searches_limit=50
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/api/login", response_model=Token)
async def login(user_credentials: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == user_credentials.email).first()
    if not user or not verify_password(user_credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

# User profile endpoint
@app.get("/api/profile")
async def get_profile(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return {
        "email": current_user.email,
        "subscription_tier": current_user.subscription_tier,
        "subscription_status": current_user.subscription_status,
        "searches_used": current_user.searches_used,
        "searches_limit": current_user.searches_limit,
        "created_at": current_user.created_at
    }

# Search limit checking for premium users
def check_premium_search_limit(user: User) -> dict:
    """Check if user has exceeded their search limit based on subscription tier"""
    if user.subscription_tier == "premium":
        return {"exceeded": False, "searches_used": user.searches_used, "unlimited": True}
    
    if user.searches_used >= user.searches_limit:
        return {"exceeded": True, "searches_used": user.searches_used, "limit": user.searches_limit}
    
    return {"exceeded": False, "searches_used": user.searches_used, "limit": user.searches_limit}

def increment_premium_search_count(user: User, db: Session):
    """Increment user search count"""
    user.searches_used += 1
    db.commit()

# Stripe webhook for subscription management
@app.post("/api/webhook/stripe")
async def stripe_webhook(request: Request, db: Session = Depends(get_db)):
    payload = await request.body()
    sig_header = request.headers.get('stripe-signature')
    
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, os.getenv("STRIPE_WEBHOOK_SECRET")
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError as e:
        raise HTTPException(status_code=400, detail="Invalid signature")
    
    if event['type'] == 'customer.subscription.created':
        subscription = event['data']['object']
        customer_id = subscription['customer']
        user = db.query(User).filter(User.stripe_customer_id == customer_id).first()
        if user:
            user.stripe_subscription_id = subscription['id']
            user.subscription_status = 'active'
            if subscription['items']['data'][0]['price']['id'] == os.getenv("STRIPE_PREMIUM_PRICE_ID"):
                user.subscription_tier = 'premium'
                user.searches_limit = 999999  # Unlimited
            db.commit()
    
    elif event['type'] == 'customer.subscription.deleted':
        subscription = event['data']['object']
        user = db.query(User).filter(User.stripe_subscription_id == subscription['id']).first()
        if user:
            user.subscription_status = 'cancelled'
            user.subscription_tier = 'basic'
            user.searches_limit = 50
            db.commit()
    
    return {"status": "success"}

# Create checkout session for subscription
@app.post("/api/create-checkout-session")
async def create_checkout_session(tier: str, current_user: User = Depends(get_current_user)):
    if tier not in ["basic", "premium"]:
        raise HTTPException(status_code=400, detail="Invalid tier")
    
    price_id = os.getenv("STRIPE_BASIC_PRICE_ID") if tier == "basic" else os.getenv("STRIPE_PREMIUM_PRICE_ID")
    
    try:
        checkout_session = stripe.checkout.Session.create(
            customer_email=current_user.email,
            line_items=[{
                'price': price_id,
                'quantity': 1,
            }],
            mode='subscription',
            success_url='https://admin.claimsafer.com/success',
            cancel_url='https://admin.claimsafer.com/cancel',
        )
        return {"url": checkout_session.url}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Now include all the existing functionality but with authentication
# Load data and setup
print("📊 Loading data...")
df = pd.read_csv("Masterfile claims with categories - Masterfile claims with categories.csv.csv")
print(f"✅ Loaded {len(df)} rows of data")

# Load GPT variations
with open("gpt_claim_variations.json", "r") as f:
    gpt_variations = json.load(f)

# Setup templates
templates = Jinja2Templates(directory="templates")

# Static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Icons
icon_claim_category = """<svg class="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
    <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
</svg>"""

icon_allowed_claims = """<svg class="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
    <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"/>
</svg>"""

icon_dosage = """<svg class="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
    <path fill-rule="evenodd" d="M3 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm0 4a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1z" clip-rule="evenodd"/>
</svg>"""

icon_pending = """<svg class="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
    <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z" clip-rule="evenodd"/>
</svg>"""

icon_notes = """<svg class="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
    <path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clip-rule="evenodd"/>
</svg>"""

# Helper functions
def normalize(name: str) -> str:
    return u_normalize('NFD', name).encode('ascii', 'ignore').decode('utf-8')

def normalize_text(s: str) -> str:
    return re.sub(r'[^\w\s]', '', s.lower())

def get_variations_for_claim(claim: str):
    """Get variations for a claim based on subscription tier"""
    if claim in gpt_variations:
        return gpt_variations[claim]
    return []

def clean_claim(c: str) -> str:
    return c.strip().lower()

def split_claims(raw) -> list[str]:
    if pd.isna(raw) or raw == "":
        return []
    claims = str(raw).split(";")
    return [clean_claim(claim) for claim in claims if clean_claim(claim)]

def tokenize(s: str) -> list[str]:
    return re.findall(r'\b\w+\b', s.lower())

def stems_of(words: list[str]) -> set[str]:
    stemmer = SnowballStemmer('english')
    return {stemmer.stem(word) for word in words}

def claim_stems(text: str) -> set[str]:
    return stems_of(tokenize(text))

def assign_best_category_from_stems(stems: set[str]) -> str:
    # This would need to be implemented based on your category logic
    return "General"

def category_for_query(query: str) -> str:
    stems = claim_stems(query)
    return assign_best_category_from_stems(stems)

def section(title, content, icon_svg, ingredient=None):
    if not content or content == "" or content == "nan":
        content = "N/A"
    
    return f"""
    <div class="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div class="flex items-center mb-4">
            <div class="flex-shrink-0">
                <div class="w-8 h-8 bg-indigo-100 rounded-lg flex items-center justify-center">
                    {icon_svg}
                </div>
            </div>
            <div class="ml-3">
                <h3 class="text-lg font-medium text-gray-900">{title}</h3>
            </div>
        </div>
        <div class="prose prose-sm text-gray-600 max-w-none">
            {content}
        </div>
    </div>
    """

def render_claim_card_collapsible(title, claims, dosage, idx, add_rewrite=True, icon_html=""):
    if not claims:
        claims = ["No claims found"]
    
    claims_html = ""
    for i, claim in enumerate(claims):
        if isinstance(claim, str) and claim.strip():
            claims_html += f'<li class="mb-2">{claim.strip()}</li>'
    
    if not claims_html:
        claims_html = '<li>No claims found</li>'
    
    rewrite_button = ""
    if add_rewrite and len(claims) > 0:
        first_claim = claims[0] if isinstance(claims[0], str) else str(claims[0])
        rewrite_button = f"""
        <button onclick="rewriteClaim('{first_claim.replace("'", "\\'")}')" 
                class="mt-3 inline-flex items-center px-3 py-1.5 border border-transparent text-xs font-medium rounded text-indigo-700 bg-indigo-100 hover:bg-indigo-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500">
            <svg class="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"/>
            </svg>
            Rewrite Claim
        </button>
        """
    
    return f"""
    <div class="bg-white rounded-lg shadow-sm border border-gray-200">
        <div class="px-6 py-4 border-b border-gray-200">
            <div class="flex items-center justify-between">
                <div class="flex items-center">
                    <div class="flex-shrink-0">
                        <div class="w-8 h-8 bg-green-100 rounded-lg flex items-center justify-center">
                            {icon_html}
                        </div>
                    </div>
                    <div class="ml-3">
                        <h3 class="text-lg font-medium text-gray-900">{title}</h3>
                    </div>
                </div>
                <button onclick="toggleCollapse('claims-{idx}')" class="text-gray-400 hover:text-gray-600">
                    <svg class="w-5 h-5 transform transition-transform duration-200" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/>
                    </svg>
                </button>
            </div>
        </div>
        <div id="claims-{idx}" class="px-6 py-4">
            <ul class="list-disc list-inside space-y-1 text-gray-700">
                {claims_html}
            </ul>
            {rewrite_button}
        </div>
    </div>
    """

# Premium search endpoints with authentication
@app.get("/", response_class=HTMLResponse)
async def premium_home(request: Request):
    """Premium home page with login/register forms"""
    return templates.TemplateResponse("premium_home.html", {"request": request})

@app.post("/search-by-ingredient-premium", response_class=HTMLResponse)
async def search_by_ingredient_premium(
    ingredient: str = Form(...), 
    country: str = Form(...), 
    request: Request = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Premium ingredient search with tier-based limits"""
    try:
        # Check premium search limit
        search_limit = check_premium_search_limit(current_user)
        
        if search_limit["exceeded"]:
            tier_name = "Premium" if current_user.subscription_tier == "premium" else "Basic"
            limit_text = "unlimited" if search_limit.get("unlimited") else f"{search_limit['limit']}"
            
            return HTMLResponse(
                f"""
                <div class="bg-red-50 border border-red-200 rounded-lg p-6 mb-6">
                    <div class="flex">
                        <div class="flex-shrink-0">
                            <svg class="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                                <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd" />
                            </svg>
                        </div>
                        <div class="ml-3">
                            <h3 class="text-sm font-medium text-red-800">
                                Search Limit Reached
                            </h3>
                            <div class="mt-2 text-sm text-red-700">
                                <p>You have used {search_limit["searches_used"]} of {limit_text} searches in your {tier_name} plan.</p>
                                <p class="mt-2">Upgrade to Premium for unlimited searches.</p>
                                <button onclick="upgradeToPremium()" class="mt-3 inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700">
                                    Upgrade to Premium
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
                """,
                status_code=403
            )
        
        # Perform the search (same logic as original)
        print(f"🔍 Premium search for ingredient: '{ingredient}' in country: '{country}' by user: {current_user.email}")
        
        # Check if 'Ingredient' column exists
        if 'Ingredient' not in df.columns:
            return HTMLResponse(
                "<p class='text-gray-600'>Error: Ingredient column not found in data.</p>",
                status_code=200
            )
        
        # Get rows for this ingredient and specific country
        matches = df[(df["Ingredient"].str.lower() == ingredient.lower()) & 
                    (df["Country"].str.lower() == country.lower())]
        
        if matches.empty:
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
            # Try multiple possible claim columns
            claim_sources = [
                row.get("Claim", ""),
                row.get("Allowed Claims", ""),
                row.get("Claims", "")
            ]
            
            for claim_source in claim_sources:
                if claim_source and str(claim_source).strip() and str(claim_source).strip() != "nan":
                    all_claims.add(str(claim_source).strip())
                    break
            
            dosage = row.get("Dosage", "")
            if dosage and str(dosage).strip() and str(dosage).strip() != "nan":
                all_dosages.add(str(dosage).strip())
                
            category = row.get("Categories", "")
            if category and str(category).strip() and str(category).strip() != "nan":
                all_categories.add(str(category).strip())
                
            pending = row.get("Health claim pending European authorisation", "")
            if pending and str(pending).strip() and str(pending).strip() != "nan":
                all_pending.add(str(pending).strip())
                
            notes = row.get("Claim Use Notes", "")
            if notes and str(notes).strip() and str(notes).strip() != "nan":
                all_notes.add(str(notes).strip())

        if not all_claims:
            return HTMLResponse(
                "<p class='text-gray-600'>No claims found for this ingredient.</p>",
                status_code=200
            )

        # Since we're already filtering by country, we don't need to group by country
        parts = [f"<h2 class='text-2xl font-bold text-gray-800 mb-6'>{ingredient} — {country}</h2>"]
        
        # Get data for the specific country
        country_claims = []
        country_dosages = set()
        country_pending = ""
        country_notes = ""
        country_categories = set()
        
        for _, row in matches.iterrows():
            # Collect claims for this country
            claim = row.get("Claim", "")
            if claim and str(claim).strip() and str(claim).strip() != "nan":
                country_claims.append(str(claim).strip())
            
            # Get country-specific dosage
            dosage = row.get("Dosage", "")
            if dosage and str(dosage).strip() and str(dosage).strip() != "nan":
                dosage_text = str(dosage).strip()
                if dosage_text.lower() != "banned":
                    country_dosages.add(dosage_text)
            
            # Get country-specific pending claims
            pending = row.get("Health claim pending European authorisation", "")
            if pending and str(pending).strip() and str(pending).strip() != "nan":
                country_pending = str(pending).strip()
            
            # Get country-specific notes
            notes = row.get("Claim Use Notes", "")
            if notes and str(notes).strip() and str(notes).strip() != "nan":
                country_notes = str(notes).strip()
            
            # Get country-specific categories
            category = row.get("Categories", "")
            if category and str(category).strip() and str(category).strip() != "nan":
                country_categories.add(str(category).strip())
        
        # Format categories for this country
        formatted_categories_html = "N/A"
        if country_categories:
            category_tags = []
            for cat in sorted(country_categories):
                category_tags.append(f'<span class="inline-block bg-indigo-100 text-indigo-800 text-sm font-medium px-3 py-1 rounded-full mr-2 mb-2">{cat}</span>')
            formatted_categories_html = "".join(category_tags)
        
        # Format dosages for this country
        country_dosage_text = ""
        if country_dosages:
            dosage_list = sorted(list(country_dosages))
            country_dosage_text = "\n".join(dosage_list)
        
        # Create sections for the specific country
        country_parts = [
            section("Claim Category", formatted_categories_html, icon_claim_category),
            render_claim_card_collapsible(
                "Allowed Claims",
                country_claims,
                "",
                1,
                add_rewrite=True,
                icon_html=icon_allowed_claims
            ),
            section("Dosage", country_dosage_text, icon_dosage),
            section("Health Claim Pending European Authorisation", country_pending, icon_pending),
            section("Claim Use Notes", country_notes, icon_notes),
        ]
        
        parts.extend(country_parts)

        html_content = "<div class='space-y-6'>" + "".join(parts) + "</div>"
        
        # Increment search count for premium user
        increment_premium_search_count(current_user, db)
        
        return HTMLResponse(html_content, status_code=200)

    except Exception as e:
        import traceback
        print(f"❌ Error in premium search_by_ingredient: {e}")
        traceback.print_exc()
        return HTMLResponse(f"<p class='text-red-600'>Error: {str(e)}</p>", status_code=500)

@app.post("/search-by-claim-premium", response_class=HTMLResponse)
async def search_by_claim_premium(
    claim: str = Form(""),
    country: str = Form(...),
    category: Optional[str] = Form(None),
    request: Request = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Premium claim search with tier-based limits and variation limits"""
    try:
        # Check premium search limit
        search_limit = check_premium_search_limit(current_user)
        
        if search_limit["exceeded"]:
            tier_name = "Premium" if current_user.subscription_tier == "premium" else "Basic"
            limit_text = "unlimited" if search_limit.get("unlimited") else f"{search_limit['limit']}"
            
            return HTMLResponse(
                f"""
                <div class="bg-red-50 border border-red-200 rounded-lg p-6 mb-6">
                    <div class="flex">
                        <div class="flex-shrink-0">
                            <svg class="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
                                <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd" />
                            </svg>
                        </div>
                        <div class="ml-3">
                            <h3 class="text-sm font-medium text-red-800">
                                Search Limit Reached
                            </h3>
                            <div class="mt-2 text-sm text-red-700">
                                <p>You have used {search_limit["searches_used"]} of {limit_text} searches in your {tier_name} plan.</p>
                                <p class="mt-2">Upgrade to Premium for unlimited searches.</p>
                                <button onclick="upgradeToPremium()" class="mt-3 inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700">
                                    Upgrade to Premium
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
                """,
                status_code=403
            )
        
        print(f"🔍 Premium search for claim: '{claim}' in country: '{country}' by user: {current_user.email}")
        
        # Get variations based on subscription tier
        variations = get_variations_for_claim(claim)
        
        # Limit variations based on subscription tier
        if current_user.subscription_tier == "basic":
            # Basic users only see 3 variations
            variations = variations[:3] if variations else []
        # Premium users see all variations (no limit)
        
        # Perform the search (same logic as original)
        # ... (implement the claim search logic here)
        
        # For now, return a simple response
        html_content = f"""
        <div class="space-y-6">
            <h2 class="text-2xl font-bold text-gray-800 mb-6">Claim Search Results</h2>
            <div class="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                <h3 class="text-lg font-medium text-gray-900 mb-4">Claim: {claim}</h3>
                <p class="text-gray-600">Country: {country}</p>
                <p class="text-gray-600">Subscription Tier: {current_user.subscription_tier.title()}</p>
                <p class="text-gray-600">Variations Found: {len(variations)}</p>
            </div>
        </div>
        """
        
        # Increment search count for premium user
        increment_premium_search_count(current_user, db)
        
        return HTMLResponse(html_content, status_code=200)

    except Exception as e:
        import traceback
        print(f"❌ Error in premium search_by_claim: {e}")
        traceback.print_exc()
        return HTMLResponse(f"<p class='text-red-600'>Error: {str(e)}</p>", status_code=500)

# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "premium"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 