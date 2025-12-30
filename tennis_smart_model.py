import os
import sys
import json
import io
import time
import random
import requests
import warnings
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from flask import Flask
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from fuzzywuzzy import fuzz

# --- FLASK APP SETUP (Renderiä varten) ---
app = Flask(__name__)
warnings.filterwarnings("ignore")

# --- KONFIGURAATIO ---
SIMULATIONS = 10000  # Monte Carlo iteraatiot per ottelu
SHEET_NAME = "Latest_ATP_Predictions" # Google Sheetin nimi
SACKMANN_BASE_URL = "https://raw.githubusercontent.com/Tennismylife/TML-Database/master"

# ========================================================
# 1. DATA INGESTION (Sackmann & ESPN)
# ========================================================

def get_atp_data():
    """Lataa ja yhdistää kuluvan ja edellisen vuoden datan (TML-Repo format)."""
    print("Step 1: Downloading Historical Data (Tennismylife Repo)...")
    current_year = datetime.now().year
    # Haetaan 2024 ja 2025
    years = [current_year - 1, current_year]
    
    frames = []
    for y in years:
        # KORJAUS: Tennismylife-repo käyttää tiedostonimeä "2025.csv", ei "atp_matches_2025.csv"
        url = f"{SACKMANN_BASE_URL}/{y}.csv"
        try:
            print(f"   -> Fetching {y} data from {url}...")
            # Lisätään timeout ja virheenkäsittely
            df = pd.read_csv(url, on_bad_lines='skip', encoding='ISO-8859-1')
            
            # Varmistetaan että päivämäärät luetaan oikein
            if 'tourney_date' in df.columns:
                df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d', errors='coerce')
                frames.append(df)
            else:
                print(f"      ! Warning: 'tourney_date' column missing in {y} data.")
                
        except Exception as e:
            print(f"      ! Warning: Could not fetch {y} data: {e}")
    
    if not frames:
        return pd.DataFrame()
    
    full_data = pd.concat(frames, ignore_index=True)
    return full_data

def get_espn_schedule():
    """Hakee ATP-otteluohjelman. Etsii seuraavat 14 päivää, kunnes löytää pelejä."""
    print("Step 2: Fetching Schedule (Smart Scan)...")
    base_url = "http://site.api.espn.com/apis/site/v2/sports/tennis/atp/scoreboard"
    
    # Etsitään max 14 päivää eteenpäin
    for i in range(14):
        search_date = datetime.now() + timedelta(days=i)
        date_str = search_date.strftime("%Y%m%d")
        display_date = search_date.strftime("%Y-%m-%d")
        
        # ESPN API:lle annetaan päivämäärä parametrina ?dates=YYYYMMDD
        url = f"{base_url}?dates={date_str}"
        
        try:
            print(f"   -> Checking date: {display_date}...")
            resp = requests.get(url).json()
            events = resp.get('events', [])
            
            matches = []
            valid_games_found = False

            for event in events:
                status = event.get('status', {}).get('type', {}).get('state')
                if status == 'post': continue # Ohitetaan pelatut
                
                # --- KORJAUS: Turvatarkistus ---
                competitions = event.get('competitions', [])
                if not competitions: 
                    # Jos lista on tyhjä, hypätään yli (estää index out of range -virheen)
                    continue 
                
                competitors = competitions[0].get('competitors', [])
                # -------------------------------

                if len(competitors) != 2: continue
                
                p1 = competitors[0]
                p2 = competitors[1]
                
                # Tournament Info
                season = event.get('season', {}).get('year')
                tourney_name = event.get('season', {}).get('slug', 'Unknown')
                
                # Yritetään kaivaa alusta (Surface)
                surface = "Hard"
                if "clay" in tourney_name.lower(): surface = "Clay"
                elif "grass" in tourney_name.lower(): surface = "Grass"
                
                matches.append({
                    'date': event.get('date'),
                    'tournament': tourney_name,
                    'surface': surface,
                    'p1_name': p1['team']['displayName'],
                    'p2_name': p2['team']['displayName'],
                    'p1_id': p1.get('id'),
                    'p2_id': p2.get('id')
                })
                valid_games_found = True
            
            if valid_games_found:
                print(f"   -> Found {len(matches)} matches on {display_date}!")
                return matches
                
        except Exception as e:
            # Tulostetaan virhe, mutta ei kaaduta (continue jatkaa seuraavaan päivään)
            print(f"      ! Warning checking {display_date}: {e}")
            continue
    
    print("   -> No upcoming matches found in the next 14 days.")
    return []
# ========================================================
# 2. FEATURE ENGINEERING
# ========================================================

def normalize_name(name):
    """Normalisoi nimen muotoon 'Sinner J.' tai 'Djokovic N.' vertailua varten."""
    parts = name.split()
    if not parts: return ""
    last_name = parts[-1]
    first_initial = parts[0][0]
    return f"{last_name} {first_initial}."

def calculate_player_stats(player_name, surface, history_df):
    """Laskee pelaajan syöttö- ja palautusprosentit viimeiseltä 52 viikolta (spesifillä alustalla)."""
    
    # Sackmannin datassa nimet on muodossa "Lastname F." tai "Lastname F. M."
    # Tehdään kevyt normalisointi
    
    # 1. Filtteröidään data (Viimeiset 52vko + Oikea alusta)
    one_year_ago = datetime.now() - timedelta(weeks=52)
    
    # Etsitään pelaajan ottelut (voittajana tai häviäjänä)
    # Käytetään osittaista string matchia koska nimiformaatit vaihtelee
    p_last = player_name.split()[-1].lower()
    
    relevant_matches = history_df[
        (history_df['tourney_date'] >= one_year_ago) &
        (history_df['surface'].str.lower() == surface.lower()) &
        ((history_df['winner_name'].str.lower().str.contains(p_last)) | 
         (history_df['loser_name'].str.lower().str.contains(p_last)))
    ].copy()
    
    if relevant_matches.empty:
        # Jos ei dataa alustalta, otetaan kaikki alustat backupina
        relevant_matches = history_df[
            (history_df['tourney_date'] >= one_year_ago) &
            ((history_df['winner_name'].str.lower().str.contains(p_last)) | 
             (history_df['loser_name'].str.lower().str.contains(p_last)))
        ].copy()
        
    if relevant_matches.empty:
        return None # Ei dataa

    # Lasketaan statsit
    serve_points_won = 0
    serve_points_total = 0
    return_points_won = 0
    return_points_total = 0
    minutes_played_recent = 0
    
    today = datetime.now()

    for _, row in relevant_matches.iterrows():
        is_winner = p_last in row['winner_name'].lower()
        
        # --- Stats Calculations ---
        # w_svpt = winner serve points, w_1stWon + w_2ndWon = points won on serve
        # l_svpt = loser serve points...
        
        if is_winner:
            s_played = row['w_svpt']
            s_won = row['w_1stWon'] + row['w_2ndWon']
            r_played = row['l_svpt']
            r_won = row['l_svpt'] - (row['l_1stWon'] + row['l_2ndWon']) # Points opponent lost on serve
            
            # Fatigue calculation (winner)
            match_date = row['tourney_date']
            if (today - match_date).days <= 7:
                minutes_played_recent += row['minutes'] if not pd.isna(row['minutes']) else 90
        else:
            s_played = row['l_svpt']
            s_won = row['l_1stWon'] + row['l_2ndWon']
            r_played = row['w_svpt']
            r_won = row['w_svpt'] - (row['w_1stWon'] + row['w_2ndWon'])
            
            # Fatigue calculation (loser)
            match_date = row['tourney_date']
            if (today - match_date).days <= 7:
                minutes_played_recent += row['minutes'] if not pd.isna(row['minutes']) else 90

        if pd.notna(s_played) and pd.notna(s_won):
            serve_points_total += s_played
            serve_points_won += s_won
        
        if pd.notna(r_played) and pd.notna(r_won):
            return_points_total += r_played
            return_points_won += r_won

    # Averages
    srv_pct = (serve_points_won / serve_points_total) if serve_points_total > 0 else 0.64 # ATP Avg approx
    ret_pct = (return_points_won / return_points_total) if return_points_total > 0 else 0.36
    
    return {
        'serve_pct': srv_pct,
        'return_pct': ret_pct,
        'recent_minutes': minutes_played_recent
    }

def get_h2h_edge(p1_name, p2_name, history_df):
    """Tarkistaa H2H tilanteen viimeiseltä 12 kuukaudelta."""
    one_year_ago = datetime.now() - timedelta(weeks=52)
    p1_last = p1_name.split()[-1].lower()
    p2_last = p2_name.split()[-1].lower()
    
    matches = history_df[
        (history_df['tourney_date'] >= one_year_ago) &
        (
            ((history_df['winner_name'].str.lower().str.contains(p1_last)) & (history_df['loser_name'].str.lower().str.contains(p2_last))) |
            ((history_df['winner_name'].str.lower().str.contains(p2_last)) & (history_df['loser_name'].str.lower().str.contains(p1_last)))
        )
    ]
    
    if matches.empty: return 0
    
    p1_wins = matches['winner_name'].str.lower().str.contains(p1_last).sum()
    total = len(matches)
    win_pct = p1_wins / total
    
    if win_pct > 0.66: return 0.02 # P1 Edge
    if win_pct < 0.34: return -0.02 # P2 Edge
    return 0

# ========================================================
# 3. MONTE CARLO CORE (Hierarchical Simulation)
# ========================================================

def simulate_match(p1_serve_prob, p2_serve_prob, iterations=10000, best_of=3):
    p1_match_wins = 0
    
    # Pre-calculate probabilities to save time in loop
    # Mutta tässä simulaatiossa arvotaan joka piste, joten silmukka on välttämätön
    
    for _ in range(iterations):
        p1_sets = 0
        p2_sets = 0
        sets_to_win = 2 if best_of == 3 else 3
        
        while p1_sets < sets_to_win and p2_sets < sets_to_win:
            # Simulate Set
            p1_games = 0
            p2_games = 0
            
            while True:
                # Check for Set Win
                if (p1_games >= 6 and p1_games - p2_games >= 2) or p1_games == 7:
                    p1_sets += 1
                    break
                if (p2_games >= 6 and p2_games - p1_games >= 2) or p2_games == 7:
                    p2_sets += 1
                    break
                
                # Tie-Break at 6-6
                if p1_games == 6 and p2_games == 6:
                    # Tie-break logic (First to 7, win by 2)
                    p1_tb = 0
                    p2_tb = 0
                    server_is_p1 = True # Foren yksinkertaistus: vuoro vaihtuu, mutta tässä keskiarvoistetaan
                    while True:
                        if (p1_tb >= 7 and p1_tb - p2_tb >= 2):
                            p1_games = 7; break
                        if (p2_tb >= 7 and p2_tb - p1_tb >= 2):
                            p2_games = 7; break
                        
                        # Tiebreak serve prob (averaged slightly or alternating)
                        prob = p1_serve_prob if server_is_p1 else (1 - p2_serve_prob)
                        if random.random() < prob:
                            if server_is_p1: p1_tb += 1
                            else: p1_tb += 1 # P1 wins return point
                        else:
                            if server_is_p1: p2_tb += 1
                            else: p2_tb += 1
                        
                        # Superyksinkertainen syötönvaihto simulaatioon (joka 2. piste)
                        if (p1_tb + p2_tb) % 2 != 0: 
                            server_is_p1 = not server_is_p1
                            
                else:
                    # Regular Game
                    # Who is serving? (Vuorotellen)
                    is_p1_serving = ((p1_games + p2_games) % 2 == 0)
                    
                    p1_pts = 0
                    p2_pts = 0
                    
                    while True:
                        # Game win condition (4 pts + 2 margin)
                        if p1_pts >= 4 and p1_pts - p2_pts >= 2:
                            p1_games += 1; break
                        if p2_pts >= 4 and p2_pts - p1_pts >= 2:
                            p2_games += 1; break
                            
                        # Point Simulation
                        prob = p1_serve_prob if is_p1_serving else (1 - p2_serve_prob)
                        
                        if random.random() < prob:
                            # Server wins point
                            if is_p1_serving: p1_pts += 1
                            else: p1_pts += 1 # P1 wins return point
                        else:
                            # Returner wins point
                            if is_p1_serving: p2_pts += 1
                            else: p2_pts += 1
        
        if p1_sets == sets_to_win:
            p1_match_wins += 1
            
    return p1_match_wins / iterations

# ========================================================
# 4. MAIN LOGIC
# ========================================================

def run_tennis_analysis():
    print(f"\n--- ATP TENNIS PREDICTION ENGINE V2.0 ---\nSimulations: {SIMULATIONS}")
    
    # 1. Load Data
    history_df = get_atp_data()
    schedule = get_espn_schedule()
    
    if history_df.empty or not schedule:
        return "Error: Data unavailable."
    
    results = []
    
    # Calculate ATP Global Averages (Baseline)
    # Normaalisti n. 64% syöttövoitto
    AVG_SERVE_PCT = 0.64 
    
    for match in schedule:
        p1 = match['p1_name']
        p2 = match['p2_name']
        surface = match['surface']
        
        print(f"Analyzing: {p1} vs {p2} ({surface})...")
        
        # 2. Get Stats
        stats1 = calculate_player_stats(p1, surface, history_df)
        stats2 = calculate_player_stats(p2, surface, history_df)
        
        if not stats1 or not stats2:
            print("   -> Skipping: Insufficient historical data.")
            continue
            
        # 3. Apply Modifiers (SPECS 3.2 & 3.3)
        # Fatigue
        fatigue_alert = ""
        if stats1['recent_minutes'] > 180:
            stats1['serve_pct'] *= 0.92
            fatigue_alert = f"Warning: {p1} Tired"
        if stats2['recent_minutes'] > 180:
            stats2['serve_pct'] *= 0.92
            fatigue_alert += f" {p2} Tired"
            
        # H2H Edge
        h2h_impact = get_h2h_edge(p1, p2, history_df)
        h2h_str = "-"
        if h2h_impact > 0: 
            h2h_str = f"{p1} H2H Edge"
        elif h2h_impact < 0:
            h2h_str = f"{p2} H2H Edge"
            
        # 4. Calculate Point Probabilities (SPECS 4.1)
        # P_srv = Avg + (A_serve - Avg) - (B_return - Avg) + H2H
        
        # P1 syöttää
        p1_srv_prob = AVG_SERVE_PCT + (stats1['serve_pct'] - AVG_SERVE_PCT) - (stats2['return_pct'] - (1-AVG_SERVE_PCT))
        p1_srv_prob += h2h_impact
        
        # P2 syöttää
        p2_srv_prob = AVG_SERVE_PCT + (stats2['serve_pct'] - AVG_SERVE_PCT) - (stats1['return_pct'] - (1-AVG_SERVE_PCT))
        p2_srv_prob -= h2h_impact # Huom: H2H toimii toisinpäin vastustajalle
        
        # Clamp probs
        p1_srv_prob = max(0.40, min(0.85, p1_srv_prob))
        p2_srv_prob = max(0.40, min(0.85, p2_srv_prob))
        
        # 5. Run Monte Carlo
        p1_win_prob = simulate_match(p1_srv_prob, p2_srv_prob, SIMULATIONS)
        
        # Output formatting
        favorite = p1 if p1_win_prob >= 0.5 else p2
        underdog = p2 if p1_win_prob >= 0.5 else p1
        win_prob = p1_win_prob if p1_win_prob >= 0.5 else (1 - p1_win_prob)
        fair_odds = 1 / win_prob if win_prob > 0 else 0
        
        results.append({
            'Date': match['date'][:10],
            'Tournament': match['tournament'],
            'Surface': surface,
            'Favorite': favorite,
            'Underdog': underdog,
            'Win Probability %': win_prob, # pidetään numerona lajittelua varten
            'Fair Odds': round(fair_odds, 2),
            'Fatigue Alert': fatigue_alert,
            'H2H Edge': h2h_str
        })
        
    # 6. Export to Google Sheets
    if results:
        df = pd.DataFrame(results)
        # Sort Highest Probability first
        df = df.sort_values('Win Probability %', ascending=False)
        
        # Format percentage string for output
        df['Win Probability %'] = (df['Win Probability %'] * 100).round(1).astype(str) + "%"
        
        print("   -> Connecting to Google Sheets...")
        try:
            json_creds = os.environ.get('GOOGLE_CREDENTIALS')
            creds_dict = json.loads(json_creds)
            scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
            creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
            client = gspread.authorize(creds)
            
            # Avaa sheet. Jos ei löydy, luo uusi tai heitä virhe.
            try:
                sheet = client.open(SHEET_NAME).sheet1
            except:
                sheet = client.create(SHEET_NAME).sheet1
                
            sheet.clear()
            sheet.update([df.columns.values.tolist()] + df.values.tolist())
            
            # Header formatting
            sheet.format('A1:I1', {'textFormat': {'bold': True}, 'backgroundColor': {'red': 0.2, 'green': 0.6, 'blue': 0.2}, 'textFormat': {'foregroundColor': {'red': 1, 'green': 1, 'blue': 1}}})
            
            return "Success: Tennis Predictions Updated!"
        except Exception as e:
            print(f"Sheets Error: {e}")
            return f"Error: {e}"
            
    return "No matches analyzed."

# --- WEB INTERFACE ---
@app.route('/')
def index():
    return "<h1>ATP Tennis Engine V2.0</h1><a href='/run'>Run Simulations</a>"

@app.route('/run')
def trigger_run():
    log_capture = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = log_capture
    
    status = "Unknown"
    try:
        status = run_tennis_analysis()
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        status = "Failed"
    finally:
        sys.stdout = original_stdout
        
    logs = log_capture.getvalue()
    return f"<h2>Status: {status}</h2><pre>{logs}</pre>"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))

    app.run(host='0.0.0.0', port=port)



