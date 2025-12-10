from __future__ import annotations

import os
from pathlib import Path
from typing import List, Literal, Optional, Dict, Any
from collections import Counter
from functools import lru_cache
import time
import requests

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timezone

from supabase import create_client, Client
from dotenv import load_dotenv

# ---------- Config ----------
APP_DIR = Path(__file__).resolve().parent  # src/services/recsys_agentic
SERVICES_DIR = APP_DIR.parent  # src/services
SRC_DIR = SERVICES_DIR.parent  # src
ROOT_DIR = SRC_DIR.parent  # project root

# Load .env.local from project root
env_path = ROOT_DIR / ".env.local"
if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ Loaded environment from: {env_path}")
else:
    print(f"⚠️ No .env.local found at: {env_path}")
    load_dotenv()  # Try to load from system env

MODEL_DIR = SERVICES_DIR / "model"

# Try multiple env var names for compatibility
VITE_SUPABASE_URL = os.getenv("VITE_SUPABASE_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_URL", "")
VITE_SUPABASE_ANON_KEY = os.getenv("VITE_SUPABASE_ANON_KEY") or os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "")  # Set to empty = disable AI insights
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")  # Optional for cloud/auth endpoints

TOP_N_DEFAULT = 5

# ---------- FastAPI ----------
app = FastAPI(title="Telvora Inference Service", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Models ----------
class AnalyticRequest(BaseModel):
    customer_id: str
    top_n: Optional[int] = TOP_N_DEFAULT

class ProductSimulationRequest(BaseModel):
    product_name: str
    category: str
    price: float
    duration_days: Optional[int] = 30

class RecommendationItem(BaseModel):
    product_id: Optional[str] = None
    product_name: str
    category: str
    price: float
    duration_days: int
    reasons: List[str]

class ChurnResult(BaseModel):
    probability: float
    label: Literal["low", "medium", "high"]
    raw_label: str

class AnalyticResponse(BaseModel):
    recommendations: Dict[str, Any]
    churn: ChurnResult
    user_category: str
    generated_at: str
    ai_insights: Optional[Dict[str, str]] = None

class ProductSimulationResponse(BaseModel):
    hits: int
    revenue: float
    conversion_rate: float
    segments: Dict[str, int]
    recommendation: str
    total_users: Optional[int] = None
    generated_at: str

# ---------- Load artifacts ----------
gemini_last_call_time = 0  # Track last API call to rate limit
gemini_call_interval = 2  # Minimum seconds between calls
global_averages: Dict[str, float] | None = None

TARGET_TO_CATEGORY_MAP = {
    'Data Booster': 'Data',
    'Voice Bundle': 'Voice',
    'Streaming Partner Pack': 'VOD',
    'Family Plan Offer': 'Combo',
    'Retention Offer': 'Combo',
    'Top-up Promo': 'Data',
    'General Offer': 'Combo',
    'Roaming Pass': 'Roaming',
    'Device Upgrade Offer': 'DeviceBundle',
}

CATEGORY_SORTING_LOGIC = {
    'Data': ('product_capacity_gb', False, 'Rekomendasi Utama (Data Kuota Terbesar):'),
    'Voice': ('product_capacity_minutes', False, 'Rekomendasi Utama (Voice Menit Terbanyak):'),
    'VOD': ('price', True, 'Rekomendasi Utama (VOD Harga Termurah):'),
    'Combo': ('price', True, 'Rekomendasi Utama (Combo Value Terbaik/Termurah):'),
    'Roaming': ('price', True, 'Rekomendasi Utama (Roaming Harga Termurah):'),
    'DeviceBundle': ('price', False, 'Rekomendasi Utama (Device Bundle Premium):'),
    'SMS': ('price_per_sms', True, 'Rekomendasi Tambahan (SMS Best Value):'),
}

CALL_THRESHOLD = 10.0
VOD_THRESHOLD = 0.4
SMS_THRESHOLD = 15.0

supabase: Optional[Client] = None
gemini_last_call_time = 0  # Track last API call to rate limit
gemini_call_interval = 2  # Minimum seconds between calls
gemini_last_call_time = 0  # Track last API call to rate limit
gemini_call_interval = 2  # Minimum seconds between calls


def fetch_all_users_paginated(page_size: int = 1000, max_attempts: int = 20) -> List[Dict[str, Any]]:
    """Fetch all customer_profile rows with pagination (same pattern as simulate_product_impact)."""
    if supabase is None:
        raise HTTPException(status_code=503, detail='Database not configured')

    all_users: List[Dict[str, Any]] = []
    page = 0
    attempt = 0
    empty_batches = 0

    while attempt < max_attempts:
        start_range = len(all_users)
        end_range = len(all_users) + page_size - 1

        res = supabase.table('customer_profile').select('*').range(start_range, end_range).execute()
        batch_size = len(res.data) if res.data else 0

        if batch_size > 0:
            all_users.extend(res.data)
            empty_batches = 0
        else:
            empty_batches += 1

        if empty_batches >= 2:
            break

        page += 1
        attempt += 1

    return all_users


def call_ollama_safe(prompt: str) -> str:
    """Call Ollama API (Cloud or Local) with rate limiting and graceful fallbacks."""
    global gemini_last_call_time

    if not OLLAMA_MODEL:
        return "AI insights tidak tersedia (OLLAMA_MODEL tidak diset)."

    try:
        # Rate limiting: minimum interval between calls
        current_time = time.time()
        time_since_last_call = current_time - gemini_last_call_time
        if time_since_last_call < gemini_call_interval:
            wait_time = gemini_call_interval - time_since_last_call
            print(f"⏳ Rate limiting: waiting {wait_time:.1f}s before next Ollama call")
            time.sleep(wait_time)

        gemini_last_call_time = time.time()

        # Prepare headers (include API key for cloud/auth endpoints)
        headers = {"Content-Type": "application/json"}
        if OLLAMA_API_KEY:
            headers["Authorization"] = f"Bearer {OLLAMA_API_KEY}"

        # Detect if using Ollama Cloud (API key present) or local
        is_cloud = OLLAMA_API_KEY and ("ollama.com" in OLLAMA_BASE_URL.lower() or "api" in OLLAMA_BASE_URL.lower())
        
        if is_cloud:
            # Ollama Cloud API endpoint format (OpenAI-compatible: /v1/chat/completions)
            base_url = OLLAMA_BASE_URL.rstrip('/')
            if base_url.endswith('/api'):
                endpoint = f"{base_url}/v1/chat/completions"
            else:
                endpoint = f"{base_url}/v1/chat/completions"
            payload = {
                "model": OLLAMA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
            }
            print(f"📡 Calling Ollama Cloud (OpenAI format): {endpoint}")
        else:
            # Local Ollama API endpoint format (uses /api/generate)
            endpoint = f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate"
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
            }
            print(f"📡 Calling Local Ollama: {endpoint}")

        resp = requests.post(endpoint, json=payload, headers=headers, timeout=60)

        if resp.status_code != 200:
            print(f"⚠️ Ollama API returned status {resp.status_code}: {resp.text[:200]}")
            return f"AI tidak tersedia (status {resp.status_code})"

        data = resp.json()
        
        # Extract text based on response format
        if is_cloud and "choices" in data:
            # OpenAI-compatible format (cloud)
            text = data["choices"][0]["message"]["content"] if data["choices"] else ""
        else:
            # Ollama format (local)
            text = data.get("response") or data.get("output") or ""
        
        return text if text else "AI tidak memberikan respons teks."
    except Exception as e:
        error_msg = str(e)
        print(f"⚠️ Ollama API error: {error_msg}")
        return f"AI tidak tersedia: {error_msg[:120]}"


def gemini_user_product_insight(pred_label: str, user_profile: Dict[str, Any], recommendations: List[RecommendationItem]) -> str:
    """Generate product recommendation insights using Gemini AI"""
    # Format recommendations for prompt
    recs_text = "\n".join([
        f"- {item.product_name} ({item.category}) - Rp {item.price:,.0f} / {item.duration_days} hari"
        for item in recommendations[:3]
    ])
    
    prompt = f"""
Role: Senior Product Marketing Analyst Telco.
Konteks: Analisis rekomendasi produk untuk pelanggan individual.

Data Pelanggan:
- Status Prediksi AI: {pred_label} (Kebutuhan utama pelanggan)
- Pengeluaran Bulanan: Rp {user_profile.get('monthly_spend', 0):,.0f}
- Penggunaan Data: {user_profile.get('avg_data_usage_gb', 0):.1f} GB/bulan
- Video Streaming: {user_profile.get('pct_video_usage', 0):.1f}%
- Durasi Panggilan: {user_profile.get('avg_call_duration', 0):.1f} menit

Produk yang Direkomendasikan:
{recs_text}

Tugas:
Buatkan deskripsi singkat (maksimal 3 kalimat) yang menjelaskan:
1. Mengapa produk-produk ini cocok untuk pelanggan berdasarkan profil penggunaan mereka
2. Benefit utama yang akan didapatkan pelanggan
3. Saran cara menawarkan produk dengan persuasif

Gunakan bahasa yang profesional namun mudah dipahami.
"""
    return call_ollama_safe(prompt)


def gemini_churn_analysis(churn_proba: float, user_profile: Dict[str, Any], pred_label: str) -> str:
    """Generate churn risk analysis using Gemini AI"""
    churn_pct = churn_proba * 100
    risk_level = "TINGGI" if churn_proba > 0.5 else "SEDANG" if churn_proba > 0.2 else "RENDAH"
    
    prompt = f"""
Role: Senior Customer Retention Analyst Telco.
Konteks: Analisis risiko churn pelanggan.

Data Pelanggan:
- Probabilitas Churn: {churn_pct:.1f}% (Risiko: {risk_level})
- Kategori Prediksi: {pred_label}
- Frekuensi Komplain: {user_profile.get('complaint_count', 0)} kali
- Pengeluaran Bulanan: Rp {user_profile.get('monthly_spend', 0):,.0f}
- Frekuensi Top-up: {user_profile.get('topup_freq', 0)}x/bulan
- Travel Score: {user_profile.get('travel_score', 0):.2f}

Tugas:
Buatkan analisis singkat (maksimal 3 kalimat) yang menjelaskan:
1. Faktor-faktor utama yang mempengaruhi tingkat risiko churn ini
2. Insight tentang perilaku pelanggan yang perlu diperhatikan
3. Strategi retensi yang disarankan untuk mengurangi risiko churn

Gunakan bahasa yang profesional dan actionable.
"""
    return call_ollama_safe(prompt)


def load_artifacts() -> None:
    global clf, label_encoder, global_averages, supabase
    model_path = MODEL_DIR / "model_dokter_rf.pkl"
    le_path = MODEL_DIR / "label_encoder.pkl"
    ga_path = MODEL_DIR / "global_averages.pkl"

    if not model_path.exists() or not le_path.exists() or not ga_path.exists():
        raise RuntimeError(f"Model artifacts not found in: {MODEL_DIR}")

    clf = joblib.load(model_path)
    label_encoder = joblib.load(le_path)
    global_averages = joblib.load(ga_path)

    global supabase
    if VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY:
        supabase = create_client(VITE_SUPABASE_URL, VITE_SUPABASE_ANON_KEY)
        print(f"✅ Supabase client initialized successfully ({VITE_SUPABASE_URL[:30]}...)")
    else:
        print("⚠️ VITE_SUPABASE_URL or VITE_SUPABASE_ANON_KEY not set - database features will be limited")
        print("   Set these in .env.local file or as environment variables")
        supabase = None
    
    # Initialize Ollama (no client object needed; we call HTTP endpoint directly)
    if OLLAMA_MODEL:
        print(f"✅ Ollama client configured (model: {OLLAMA_MODEL}, base: {OLLAMA_BASE_URL})")
    else:
        print("⚠️ OLLAMA_MODEL not set - AI insights will be disabled")


def abs_if_needed(df: pd.DataFrame, col: str) -> None:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        df[col] = df[col].abs()


def prepare_features(customer_row: Dict[str, Any]) -> pd.DataFrame:
    # Build a single-row DataFrame with expected columns
    # Columns inferred from training logic
    df = pd.DataFrame([customer_row])

    # Ensure numeric conversions similar to training
    numeric_features_list = [
        'avg_data_usage_gb', 'pct_video_usage', 'avg_call_duration', 'sms_freq',
        'monthly_spend', 'topup_freq', 'travel_score', 'complaint_count'
    ]

    # Replace comma decimal and coerce
    for col in numeric_features_list:
        if col in df.columns:
            if isinstance(df[col].iloc[0], str):
                df[col] = df[col].astype(str).str.replace(',', '.')
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Apply abs fix for negatives
    abs_if_needed(df, 'avg_call_duration')
    abs_if_needed(df, 'monthly_spend')

    # Drop ID and target-like cols if present
    for drop_col in ('customer_id', 'target_offer', 'target_encoded'):
        if drop_col in df.columns:
            df = df.drop(columns=[drop_col])

    return df


def fetch_customer(customer_id: str) -> Dict[str, Any]:
    if supabase is None:
        raise HTTPException(status_code=503, detail='Database not configured. Please set VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY in .env.local')
    res = supabase.table('customer_profile').select('*').eq('customer_id', customer_id).single().execute()
    if res.data is None:
        raise HTTPException(status_code=404, detail='Customer not found')
    return res.data


def fetch_products() -> pd.DataFrame:
    if supabase is None:
        raise HTTPException(status_code=503, detail='Database not configured. Please set VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY in .env.local')
    res = supabase.table('product_catalog').select('*').execute()
    products = res.data or []
    if not products:
        raise HTTPException(status_code=404, detail='No products found')
    df = pd.DataFrame(products)
    # Add derived columns
    eps = 1e-6
    df['price_per_gb'] = df['price'] / (df.get('product_capacity_gb', 0) + eps)
    df['price_per_minute'] = df['price'] / (df.get('product_capacity_minutes', 0) + eps)
    df['price_per_sms'] = df['price'] / (df.get('product_capacity_sms', 0) + eps)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


def compute_churn_bucket(pred_label: str, user_row: Dict[str, Any], global_avgs: Dict[str, float]) -> str:
    """
    Compute churn risk bucket RULES-BASED (sesuai notebook Cell 5 logic).
    
    Rules:
    🔴 HIGH RISK:
       - Prediksi model = 'Retention Offer'
    
    🟡 MEDIUM RISK:
       - BUKAN Retention Offer
       - ADA complaint
       - MINIMAL 2 indikator usage (data/call/sms) di bawah rata-rata
       - topup_freq di bawah rata-rata
    
    🟢 LOW RISK (multiple cases):
       CASE A: no complaint + minimal 2 usage >= rata-rata
       CASE B: ada complaint + topup >= rata-rata
       CASE C: no complaint + semua usage >= rata-rata
       CASE D: no complaint + topup <= rata-rata + minimal 1 usage >= rata-rata
       CASE E: no complaint + topup >= rata-rata + semua usage >= rata-rata
    """
    # Extract thresholds dari global_averages
    avg_data = float(global_avgs.get('avg_data_usage_gb', 0))
    avg_call = float(global_avgs.get('avg_call_duration', 0))
    avg_sms = float(global_avgs.get('sms_freq', 0))
    avg_topup = float(global_avgs.get('topup_freq', 0))
    
    # Extract user values
    usage_data = float(user_row.get('avg_data_usage_gb') or 0)
    usage_call = float(user_row.get('avg_call_duration') or 0)
    usage_sms = float(user_row.get('sms_freq') or 0)
    usage_topup = float(user_row.get('topup_freq') or 0)
    complaints = float(user_row.get('complaint_count') or 0)
    
    has_complaint = complaints > 0
    
    # Below/above average masks
    below_data = usage_data < avg_data
    below_call = usage_call < avg_call
    below_sms = usage_sms < avg_sms
    below_topup = usage_topup < avg_topup
    
    aboveeq_data = usage_data >= avg_data
    aboveeq_call = usage_call >= avg_call
    aboveeq_sms = usage_sms >= avg_sms
    aboveeq_topup = usage_topup >= avg_topup
    
    # Count indicators
    num_below_usage = sum([below_data, below_call, below_sms])
    num_above_usage = sum([aboveeq_data, aboveeq_call, aboveeq_sms])
    all_usage_above = aboveeq_data and aboveeq_call and aboveeq_sms
    
    # === HIGH RISK ===
    if pred_label == 'Retention Offer':
        return 'high'
    
    # === MEDIUM RISK ===
    # NOT Retention + HAS complaint + 2+ usage below avg + topup below avg
    if (not (pred_label == 'Retention Offer') and 
        has_complaint and 
        num_below_usage >= 2 and 
        below_topup):
        return 'medium'
    
    # === LOW RISK (multiple cases) ===
    # CASE A: no complaint + minimal 2 usage >= avg
    if not has_complaint and num_above_usage >= 2:
        return 'low'
    
    # CASE B: ada complaint + topup >= avg (loyal tapi cerewet)
    if has_complaint and aboveeq_topup:
        return 'low'
    
    # CASE C: no complaint + semua usage >= avg
    if not has_complaint and all_usage_above:
        return 'low'
    
    # CASE D: no complaint + topup <= avg + minimal 1 usage >= avg
    if not has_complaint and usage_topup <= avg_topup and num_above_usage >= 1:
        return 'low'
    
    # CASE E: no complaint + topup >= avg + semua usage >= avg
    if not has_complaint and aboveeq_topup and all_usage_above:
        return 'low'
    
    # Default: MEDIUM (untuk case yang tidak masuk LOW atau HIGH)
    return 'medium'


def recommend_products(pred_label: str, user_row: Dict[str, Any], products_df: pd.DataFrame, total_recommendations: int) -> List[RecommendationItem]:
    """Notebook-inspired recommendation v16 (max wallet share, filler murah)."""
    recs: List[RecommendationItem] = []
    seen = set()

    budget = float(user_row.get('monthly_spend') or 0.0)
    primary_category = TARGET_TO_CATEGORY_MAP.get(pred_label)

    def add_items(df: pd.DataFrame, desc: str, take: int) -> None:
        nonlocal recs, seen
        for _, row in df.head(take).iterrows():
            recs.append(RecommendationItem(
                product_id=str(row.get('product_id') or ''),
                product_name=str(row.get('product_name') or ''),
                category=str(row.get('category') or ''),
                price=float(row.get('price') or 0),
                duration_days=int(row.get('duration_days') or 30),
                reasons=[desc, f"Sesuai prediksi model: {pred_label}"]
            ))
            seen.add(row.get('product_name'))

    # Primary
    if primary_category:
        base = products_df[products_df['category'] == primary_category].copy()

        if primary_category == 'Retention Offer':
            top = base[base['price'] <= budget * 0.8].sort_values('price', ascending=True)
            if top.empty:
                top = base.sort_values('price', ascending=True)
            add_items(top, "Rekomendasi Utama (Retensi Hemat):", 3)

        elif primary_category == 'DeviceBundle':
            top = base[base['price'] <= budget].sort_values('price', ascending=False)
            if top.empty:
                top = base.sort_values('price', ascending=True)
            add_items(top, "Rekomendasi Utama (Upgrade Gadget):", 3)

        else:
            budget_ok = base[base['price'] <= budget].sort_values('price', ascending=False)
            top = budget_ok if not budget_ok.empty else base.sort_values('price', ascending=True)
            add_items(top, f"Rekomendasi Utama ({primary_category} - Premium):", 3)

    remaining = total_recommendations - len(recs)

    # Secondary cross-sell murah + durasi pendek
    if remaining > 0 and global_averages is not None:
        scores = [
            ('Data', float(user_row.get('avg_data_usage_gb') or 0) / (global_averages['avg_data_usage_gb'] + 1e-6)),
            ('Voice', float(user_row.get('avg_call_duration') or 0) / (global_averages['avg_call_duration'] + 1e-6)),
            ('VOD', float(user_row.get('pct_video_usage') or 0) / (global_averages['pct_video_usage'] + 1e-6)),
            ('SMS', float(user_row.get('sms_freq') or 0) / (global_averages['sms_freq'] + 1e-6)),
        ]
        sorted_scores = sorted([s for s in scores if s[1] > 1.0], key=lambda x: x[1], reverse=True)
        if not sorted_scores and primary_category != 'Combo':
            sorted_scores.append(('Combo', 1.0))

        for cat, _ in sorted_scores:
            if remaining <= 0:
                break
            if primary_category and cat == primary_category:
                continue

            filtered = products_df[
                (products_df['category'] == cat) &
                (products_df['price'] <= budget * 0.5) &
                (~products_df['product_name'].isin(seen))
            ].copy()
            filtered['is_short'] = filtered['duration_days'].apply(lambda x: 1 if x <= 7 else 2)
            top_sec = filtered.sort_values(by=['is_short', 'price'], ascending=[True, True])

            if not top_sec.empty:
                add_items(top_sec, f"Rekomendasi Sekunder ({cat})", 1)
                remaining = total_recommendations - len(recs)

    # Fallback fillers
    if len(recs) < total_recommendations:
        remaining = total_recommendations - len(recs)
        fillers = products_df[(~products_df['product_name'].isin(seen)) & (products_df['price'] <= budget * 0.5)]
        fillers = fillers.sort_values('price', ascending=True)
        for _, row in fillers.head(remaining).iterrows():
            recs.append(RecommendationItem(
                product_id=str(row.get('product_id') or ''),
                product_name=str(row.get('product_name') or ''),
                category=str(row.get('category') or ''),
                price=float(row.get('price') or 0),
                duration_days=int(row.get('duration_days') or 30),
                reasons=[f"{row.get('category')}", "Filler value"]
            ))
            if len(recs) >= total_recommendations:
                break

    return recs[:total_recommendations]


def simulate_product_impact(new_product: Dict[str, Any]) -> Dict[str, Any]:
    """Simulate product impact across all customers (notebook logic)."""
    if supabase is None:
        raise HTTPException(status_code=503, detail='Database not configured')
    if clf is None or label_encoder is None:
        raise HTTPException(status_code=500, detail='Model not loaded')

    # Fetch ALL customers using pagination (Supabase max ~1000 per request)
    all_users = []
    page = 0
    page_size = 1000
    max_attempts = 20  # Safety limit untuk prevent infinite loop
    attempt = 0
    empty_batches = 0  # Counter untuk batches kosong berturut-turut

    while attempt < max_attempts:
        # Gunakan len(all_users) sebagai offset yang akurat (bukan page * page_size)
        # Karena data mungkin tidak selalu eksak 1000 per batch
        start_range = len(all_users)
        end_range = len(all_users) + page_size - 1
        
        print(f"   📥 Fetching batch {page + 1} (rows {start_range}-{end_range})...")
        res = supabase.table('customer_profile').select('*').range(start_range, end_range).execute()
        
        batch_size = len(res.data) if res.data else 0
        print(f"   ✅ Got {batch_size} users | Total collected: {len(all_users) + batch_size}")
        
        # Tambahkan data ke list JIKA ada
        if batch_size > 0:
            all_users.extend(res.data)
            empty_batches = 0  # Reset counter saat ada data
        else:
            empty_batches += 1
        
        # Jika dapat 2+ batches kosong berturut-turut, STOP (truly no more data)
        if empty_batches >= 2:
            print(f"   🛑 No data in {empty_batches} consecutive batches - pagination complete")
            break
        
        # Selalu lanjut ke batch berikutnya
        page += 1
        attempt += 1
    
    print(f"   ✅ Pagination complete after {page} batches")

    # Build features DataFrame properly (batch processing)
    df_users = pd.DataFrame(all_users)
    
    # Prepare features using same logic as prepare_features but for multiple rows
    numeric_features_list = [
        'avg_data_usage_gb', 'pct_video_usage', 'avg_call_duration', 'sms_freq',
        'monthly_spend', 'topup_freq', 'travel_score', 'complaint_count'
    ]
    
    X = df_users.copy()
    
    # Replace comma decimal and coerce for numeric columns
    for col in numeric_features_list:
        if col in X.columns:
            X[col] = X[col].astype(str).str.replace(',', '.')
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    
    # Apply abs fix for negatives
    if 'avg_call_duration' in X.columns:
        X['avg_call_duration'] = X['avg_call_duration'].abs()
    if 'monthly_spend' in X.columns:
        X['monthly_spend'] = X['monthly_spend'].abs()
    
    # Drop ID and target-like cols
    for drop_col in ('customer_id', 'target_offer', 'target_encoded'):
        if drop_col in X.columns:
            X = X.drop(columns=[drop_col])


    # Predict all users
    all_preds_idx = clf.predict(X)
    all_labels = label_encoder.inverse_transform(all_preds_idx)

    hits = 0
    revenue = 0.0
    segments = {}

    for i, label in enumerate(all_labels):
        budget = float(all_users[i].get('monthly_spend') or 0)
        cat = TARGET_TO_CATEGORY_MAP.get(label)
        
        if cat == new_product['category']:
            if new_product['price'] <= budget:
                hits += 1
                revenue += new_product['price']
                segments[label] = segments.get(label, 0) + 1

    total_users = len(all_users)
    conversion_rate = (hits / total_users * 100) if total_users > 0 else 0

    # Generate recommendation text
    if conversion_rate > 20:
        recommendation = f"Produk ini memiliki potensi TINGGI dengan {hits} target user ({conversion_rate:.1f}%). Revenue potensial: Rp {revenue:,.0f}. Sangat direkomendasikan untuk produksi."
    elif conversion_rate > 10:
        recommendation = f"Produk ini memiliki potensi SEDANG dengan {hits} target user ({conversion_rate:.1f}%). Revenue potensial: Rp {revenue:,.0f}. Pertimbangkan optimasi harga atau fitur."
    else:
        recommendation = f"Produk ini memiliki potensi RENDAH dengan {hits} target user ({conversion_rate:.1f}%). Revenue potensial: Rp {revenue:,.0f}. Perlu evaluasi ulang positioning produk."

    return {
        'hits': hits,
        'revenue': revenue,
        'conversion_rate': conversion_rate,
        'segments': segments,
        'recommendation': recommendation,
        'total_users': total_users
    }


@app.on_event("startup")
def _startup():
    load_artifacts()


@app.post("/infer/analytic", response_model=AnalyticResponse)
def infer_analytic(payload: AnalyticRequest):
    if clf is None or label_encoder is None or global_averages is None:
        raise HTTPException(status_code=500, detail='Artifacts not loaded')

    customer_row = fetch_customer(payload.customer_id)
    X = prepare_features(customer_row)

    # Predict label
    pred_idx = clf.predict(X)
    pred_label = label_encoder.inverse_transform(pred_idx)[0]
    user_category = TARGET_TO_CATEGORY_MAP.get(pred_label, "Unknown")

    # Churn probability based on "Retention Offer" class
    class_list = list(label_encoder.classes_)
    try:
        idx_ret = class_list.index('Retention Offer')
        proba_all = clf.predict_proba(X)
        churn_proba = float(proba_all[0][idx_ret])
    except Exception:
        churn_proba = 0.0

    churn_label = compute_churn_bucket(pred_label, customer_row, global_averages)

    # Products + recommendations
    products_df = fetch_products()
    items = recommend_products(pred_label, customer_row, products_df, payload.top_n or TOP_N_DEFAULT)

    # Generate AI insights if Gemini is available
    ai_insights = None
    if OLLAMA_MODEL:
        try:
            product_insight = gemini_user_product_insight(pred_label, customer_row, items)
            # Only call churn analysis if product insight succeeded
            if "tidak tersedia" not in product_insight.lower() and "quota" not in product_insight.lower():
                churn_insight = gemini_churn_analysis(churn_proba, customer_row, pred_label)
            else:
                churn_insight = product_insight
            
            ai_insights = {
                'product_recommendation': product_insight,
                'churn_analysis': churn_insight
            }
        except Exception as e:
            print(f"Error generating AI insights: {e}")
            # Don't fail the whole request, just skip AI insights

    return AnalyticResponse(
        recommendations={
            'topN': payload.top_n or TOP_N_DEFAULT,
            'items': [i.model_dump() for i in items],
        },
        churn=ChurnResult(probability=churn_proba, label=churn_label, raw_label=pred_label),
        user_category=user_category,
        generated_at=datetime.now(timezone.utc).isoformat(),
        ai_insights=ai_insights,
    )


@app.post("/infer/simulate-product", response_model=ProductSimulationResponse)
def simulate_product(payload: ProductSimulationRequest):
    """Simulate product impact across customer base using ML model."""
    new_product = {
        'product_name': payload.product_name,
        'category': payload.category,
        'price': payload.price,
        'duration_days': payload.duration_days or 30
    }
    
    result = simulate_product_impact(new_product)
    
    return ProductSimulationResponse(
        hits=result['hits'],
        revenue=result['revenue'],
        conversion_rate=result['conversion_rate'],
        segments=result['segments'],
        recommendation=result['recommendation'],
        total_users=result.get('total_users'),
        generated_at=datetime.now(timezone.utc).isoformat()
    )


@app.get("/analytics/overview")
def analytics_overview():
    """Global analytics: behaviour trends + product effectiveness (rule-based, no AI)."""
    if clf is None or label_encoder is None:
        raise HTTPException(status_code=500, detail='Model not loaded')

    # 1) Ambil semua user
    all_users = fetch_all_users_paginated()
    if not all_users:
        raise HTTPException(status_code=404, detail='No users found')

    # 2) Siapkan DataFrame dan features (re-use logic dari simulate_product_impact)
    df_users = pd.DataFrame(all_users)
    numeric_features_list = [
        'avg_data_usage_gb', 'pct_video_usage', 'avg_call_duration', 'sms_freq',
        'monthly_spend', 'topup_freq', 'travel_score', 'complaint_count'
    ]

    X = df_users.copy()
    for col in numeric_features_list:
        if col in X.columns:
            X[col] = X[col].astype(str).str.replace(',', '.')
            X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

    if 'avg_call_duration' in X.columns:
        X['avg_call_duration'] = X['avg_call_duration'].abs()
    if 'monthly_spend' in X.columns:
        X['monthly_spend'] = X['monthly_spend'].abs()

    for drop_col in ('customer_id', 'target_offer', 'target_encoded'):
        if drop_col in X.columns:
            X = X.drop(columns=[drop_col])

    # 3) Prediksi label
    all_preds_idx = clf.predict(X)
    all_labels = label_encoder.inverse_transform(all_preds_idx)

    total_users = len(df_users)
    
    # 3.5) Model Performance Metrics (sesuai notebook cell 9)
    model_performance = {
        'accuracy': None,
        'total_classes': len(label_encoder.classes_),
        'classes': label_encoder.classes_.tolist()
    }

    if 'target_offer' in df_users.columns:
        from sklearn.metrics import accuracy_score

        y_raw = df_users['target_offer']
        if y_raw.notna().any():
            class_set = set(label_encoder.classes_)

            # 1) String match (trim, case-sensitive to preserve class names)
            y_str = y_raw.astype(str).str.strip()
            mask_valid_str = y_str.isin(class_set)
            if mask_valid_str.any():
                acc = accuracy_score(y_str[mask_valid_str].values, np.array(all_labels)[mask_valid_str])
                model_performance['accuracy'] = round(acc * 100, 2)
            else:
                # 2) Numeric decode -> inverse label_encoder
                y_num = pd.to_numeric(y_raw, errors='coerce')
                mask_num = y_num.notna()
                if mask_num.any():
                    try:
                        y_int = y_num[mask_num].astype(int).values
                        y_true_decoded = label_encoder.inverse_transform(y_int)
                        acc = accuracy_score(y_true_decoded, np.array(all_labels)[mask_num])
                        model_performance['accuracy'] = round(acc * 100, 2)
                    except Exception:
                        model_performance['accuracy'] = None

    # 4) Behaviour trends
    high_data_threshold = None
    high_data_users = 0
    q75_usage = None
    if 'avg_data_usage_gb' in df_users.columns:
        q75_usage = df_users['avg_data_usage_gb'].quantile(0.75)
        high_data_threshold = max(10.0, q75_usage)
        high_data_users = int((df_users['avg_data_usage_gb'] >= high_data_threshold).sum())

    plan_spend = {'Prepaid': 0.0, 'Postpaid': 0.0}
    plan_counts = {'Prepaid': 0, 'Postpaid': 0}
    if 'plan_type' in df_users.columns:
        spend_by_plan = df_users.groupby('plan_type')['monthly_spend'].mean().to_dict()
        plan_spend['Prepaid'] = float(spend_by_plan.get('Prepaid', 0))
        plan_spend['Postpaid'] = float(spend_by_plan.get('Postpaid', 0))
        counts = df_users['plan_type'].value_counts().to_dict()
        plan_counts['Prepaid'] = int(counts.get('Prepaid', 0))
        plan_counts['Postpaid'] = int(counts.get('Postpaid', 0))

    video_voice = {
        'video_lovers': 0,
        'voice_lovers': 0,
        'balanced': 0,
        'median_call': 0.0
    }
    if 'pct_video_usage' in df_users.columns and 'avg_call_duration' in df_users.columns:
        video_pct = df_users['pct_video_usage'].values
        call_dur = df_users['avg_call_duration'].values
        median_call = float(np.median(call_dur)) if len(call_dur) else 0.0
        video_lovers_mask = video_pct >= 0.6
        voice_lovers_mask = (video_pct <= 0.4) & (call_dur >= median_call)
        balanced_mask = ~(video_lovers_mask | voice_lovers_mask)

        video_voice = {
            'video_lovers': int(video_lovers_mask.sum()),
            'voice_lovers': int(voice_lovers_mask.sum()),
            'balanced': int(balanced_mask.sum()),
            'median_call': median_call
        }

    # 5) Top products - OPTIMIZED: Pre-compute & vectorized operations (sesuai notebook cell 11)
    products_df = fetch_products()
    all_recs = []
    
    print(f"   📊 Menghitung top products dari {total_users} user (optimized)...")
    
    # OPTIMIZATION 1: Pre-compute products per category (sekali saja, bukan per-user)
    products_by_cat = {}
    for cat in TARGET_TO_CATEGORY_MAP.values():
        cat_products = products_df[products_df['category'] == cat].copy()
        # Pre-sort by price untuk setiap kategori
        products_by_cat[cat] = cat_products.sort_values('price', ascending=False)
    
    # OPTIMIZATION 2: Vectorized budget array
    budgets = df_users['monthly_spend'].values
    
    # OPTIMIZATION 3: Batch processing dengan minimal pandas operations
    start_time = time.time()
    
    for idx in range(total_users):
        pred_label = all_labels[idx]
        user_budget = budgets[idx]
        
        # Get primary category
        cat_primer = TARGET_TO_CATEGORY_MAP.get(pred_label)
        
        if cat_primer and cat_primer in products_by_cat:
            cat_prods = products_by_cat[cat_primer]
            
            # Quick filtering: hanya ambil produk dalam budget
            affordable = cat_prods[cat_prods['price'] <= user_budget]
            
            if not affordable.empty:
                # Ambil top 3 termahal dalam budget (sudah sorted descending)
                top_recs = affordable.head(3)['product_name'].tolist()
                all_recs.extend(top_recs)
            else:
                # Fallback: ambil yang termurah
                if not cat_prods.empty:
                    all_recs.append(cat_prods.iloc[-1]['product_name'])
        
        # Progress log setiap 2500 user
        if (idx + 1) % 2500 == 0:
            elapsed = time.time() - start_time
            print(f"   ... processed {idx + 1}/{total_users} users ({elapsed:.1f}s)")
    
    elapsed_total = time.time() - start_time
    print(f"   ✅ Completed in {elapsed_total:.2f} seconds")
    
    # Hitung top 10 produk yang paling sering direkomendasikan
    product_counter = Counter(all_recs)
    total_recs = len(all_recs)
    top_products = []
    for prod_name, cnt in product_counter.most_common(10):
        top_products.append({
            'product_name': prod_name,
            'count': cnt,
            'percentage': round((cnt / total_recs) * 100, 1) if total_recs else 0.0
        })

    return {
        'total_users': total_users,
            'model_performance': model_performance,
        'high_data': {
            'count': high_data_users,
            'percentage': round((high_data_users / total_users) * 100, 1) if total_users else 0.0,
            'threshold': high_data_threshold,
            'q75_usage': q75_usage
        },
        'plan_spend': plan_spend,
        'plan_counts': plan_counts,
        'video_voice': video_voice,
        'top_products': top_products,
        'generated_at': datetime.now(timezone.utc).isoformat()
    }


@app.get("/analytics/churn-composition")
def get_churn_composition():
    """
    Compute churn risk composition untuk ALL customers (10k users).
    Menghitung: Low Risk, Medium Risk, High Risk distribution.
    Output: data untuk pie chart dashboard.
    """
    if supabase is None:
        raise HTTPException(status_code=503, detail='Database not configured')
    if clf is None or label_encoder is None:
        raise HTTPException(status_code=500, detail='Model not loaded')

    try:
        print("⏳ Menghitung Churn Composition untuk ALL customers...")
        
        # 1. Fetch all customers with pagination (Supabase ~1000 per batch)
        all_users = []
        page = 0
        page_size = 1000
        max_attempts = 20  # Safety limit untuk prevent infinite loop
        attempt = 0
        empty_batches = 0  # Counter untuk batches kosong berturut-turut

        while attempt < max_attempts:
            # Gunakan len(all_users) sebagai offset yang akurat (bukan page * page_size)
            # Karena data mungkin tidak selalu eksak 1000 per batch
            start_range = len(all_users)
            end_range = len(all_users) + page_size - 1
            
            print(f"   📥 Fetching batch {page + 1} (rows {start_range}-{end_range})...")
            
            # JANGAN GUNAKAN .limit() - biarkan Supabase return sesuai range
            res = supabase.table('customer_profile').select('*').range(start_range, end_range).execute()
            
            batch_size = len(res.data) if res.data else 0
            print(f"   ✅ Got {batch_size} users | Total collected: {len(all_users) + batch_size}")
            
            # Tambahkan data ke list JIKA ada
            if batch_size > 0:
                all_users.extend(res.data)
                empty_batches = 0  # Reset counter saat ada data
            else:
                empty_batches += 1
            
            # Jika dapat 2+ batches kosong berturut-turut, STOP (truly no more data)
            if empty_batches >= 2:
                print(f"   🛑 No data in {empty_batches} consecutive batches - pagination complete")
                break
            
            # Selalu lanjut ke batch berikutnya
            page += 1
            attempt += 1
        
        print(f"   ✅ Pagination complete after {page} batches")

        if not all_users:
            return {
                "total_users": 0,
                "composition": {"high": {"count": 0, "percentage": 0}, "medium": {"count": 0, "percentage": 0}, "low": {"count": 0, "percentage": 0}},
                "churn_rate": 0,
                "revenue_at_risk": 0,
                "generated_at": datetime.now(timezone.utc).isoformat()
            }

        print(f"✅ Fetched {len(all_users)} customers")

        # 2. Build features DataFrame
        df_users = pd.DataFrame(all_users)
        numeric_features_list = [
            'avg_data_usage_gb', 'pct_video_usage', 'avg_call_duration', 'sms_freq',
            'monthly_spend', 'topup_freq', 'travel_score', 'complaint_count'
        ]
        
        X = df_users.copy()
        
        # Clean numeric columns
        for col in numeric_features_list:
            if col in X.columns:
                X[col] = X[col].astype(str).str.replace(',', '.')
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
        
        # Fix negatives
        if 'avg_call_duration' in X.columns:
            X['avg_call_duration'] = X['avg_call_duration'].abs()
        if 'monthly_spend' in X.columns:
            X['monthly_spend'] = X['monthly_spend'].abs()
        
        # Drop ID and target cols
        for drop_col in ('customer_id', 'target_offer', 'target_encoded'):
            if drop_col in X.columns:
                X = X.drop(columns=[drop_col])

        # 3. Predict all users
        all_preds_idx = clf.predict(X)
        all_labels = label_encoder.inverse_transform(all_preds_idx)

        # Get churn probabilities
        class_list = list(label_encoder.classes_)
        try:
            idx_ret = class_list.index('Retention Offer')
            proba_all = clf.predict_proba(X)
        except Exception:
            proba_all = None

        # 4. Compute churn risk buckets untuk ALL users menggunakan RULES-BASED logic (sesuai notebook)
        churn_buckets = {'high': 0, 'medium': 0, 'low': 0}
        user_churn_data = []
        
        for i, label in enumerate(all_labels):
            bucket = compute_churn_bucket(label, all_users[i], global_averages)
            churn_buckets[bucket] += 1
            
            # Simpan per-user data untuk revenue at risk calculation
            user_churn_data.append({
                'customer_id': all_users[i].get('customer_id'),
                'risk_level': bucket,
                'monthly_spend': float(all_users[i].get('monthly_spend') or 0)
            })

        total_users = len(all_users)
        
        # 5. Calculate percentages
        churn_rate_pct = (churn_buckets['high'] / total_users * 100) if total_users > 0 else 0.0
        medium_pct = (churn_buckets['medium'] / total_users * 100) if total_users > 0 else 0.0
        low_pct = (churn_buckets['low'] / total_users * 100) if total_users > 0 else 0.0

        # 6. Calculate revenue at risk (HIGH risk users only)
        revenue_at_risk = sum(
            item['monthly_spend']
            for item in user_churn_data 
            if item['risk_level'] == 'high'
        )

        print(f"✅ Churn Composition Analysis Results:")
        print(f"   📊 Total Users Analyzed: {total_users}")
        print(f"   🔴 High Risk: {churn_buckets['high']} users ({churn_rate_pct:.1f}%)")
        print(f"   🟡 Medium Risk: {churn_buckets['medium']} users ({medium_pct:.1f}%)")
        print(f"   🟢 Low Risk: {churn_buckets['low']} users ({low_pct:.1f}%)")
        print(f"   💰 Revenue at Risk: Rp {revenue_at_risk:,.0f}")

        return {
            'total_users': total_users,
            'composition': {
                'high': {
                    'count': int(churn_buckets['high']),
                    'percentage': round(churn_rate_pct, 1)
                },
                'medium': {
                    'count': int(churn_buckets['medium']),
                    'percentage': round(medium_pct, 1)
                },
                'low': {
                    'count': int(churn_buckets['low']),
                    'percentage': round(low_pct, 1)
                }
            },
            'churn_rate': round(churn_rate_pct, 1),
            'revenue_at_risk': round(revenue_at_risk, 0),
            'generated_at': datetime.now(timezone.utc).isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error calculating churn composition: {e}")
        raise HTTPException(status_code=500, detail=f'Error calculating churn composition: {str(e)}')


@app.get("/health")
def health():
    return {"status": "ok"}
