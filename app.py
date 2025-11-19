import streamlit as st
import pandas as pd
import joblib
import datetime

# --- 1. LOAD THE BRAINS ---
# We use @st.cache_data to make it load faster
@st.cache_data
def load_files():
    model = joblib.load("football_model.pkl")
    encoders = joblib.load("team_encoding.pkl")
    stats = joblib.load("team_stats.pkl")
    return model, encoders, stats

rf, team_ids, team_stats = load_files()

# --- 2. THE WEB INTERFACE ---
st.title("⚽ Premier League Predictor (2025/26)")
st.write("Select two teams to see who wins based on recent form.")

# Create two columns for the dropdowns
col1, col2 = st.columns(2)

with col1:
    home_team = st.selectbox("Home Team", sorted(team_ids.keys()))

with col2:
    # Filter away list so you can't select the same team twice
    away_teams = [t for t in sorted(team_ids.keys()) if t != home_team]
    away_team = st.selectbox("Away Team", away_teams, index=0)

# Optional: Pick Time and Day
match_date = st.date_input("Match Date", datetime.date(2025, 8, 12))
match_time = st.time_input("Match Time", datetime.time(15, 0))

# --- 3. PREDICTION LOGIC ---
if st.button("Predict Winner"):
    
    # A. Get the IDs
    home_code = team_ids[home_team]
    away_code = team_ids[away_team]
    
    # B. Get Time Info
    hour = match_time.hour
    day_code = match_date.weekday()
    
    # C. Get The Form (Rolling Averages)
    # We grab the latest known stats for the Home Team
    home_form = team_stats.loc[home_team].values
    
    # We grab the latest known stats for the Away Team
    # Note: We use the same 'stats' DB because form is form.
    away_form = team_stats.loc[away_team].values
    
    # Combine into one list of inputs
    # The order MUST match: [home_code, away_code, hour, day_code, ...home_rolling..., ...away_rolling...]
    input_data = [home_code, away_code, hour, day_code] + list(home_form) + list(away_form)
    
    # Create a DataFrame (names aren't strictly needed for prediction, but good for safety)
    # We just pass the list directly to the model
    prediction_probs = rf.predict_proba([input_data])[0]
    
    # --- 4. DISPLAY RESULTS ---
    # prediction_probs = [Prob Away, Prob Draw, Prob Home]
    prob_away = prediction_probs[0]
    prob_draw = prediction_probs[1]
    prob_home = prediction_probs[2]
    
    st.divider()
    
    # Calculate who is the favorite
    if prob_home > 0.60:
        st.success(f"🏆 **Prediction: {home_team} Wins!**")
        st.write(f"Confidence: **{prob_home*100:.1f}%**")
    elif prob_away > 0.60:
        st.success(f"🏆 **Prediction: {away_team} Wins!**")
        st.write(f"Confidence: **{prob_away*100:.1f}%**")
    else:
        st.warning("⚠️ **Prediction: Too Close to Call (Risky Bet)**")
        st.write("The model is not confident in a winner.")

    # Show the breakdown chart
    st.write("### Probability Breakdown")
    chart_data = pd.DataFrame({
        "Result": [f"{away_team} Win", "Draw", f"{home_team} Win"],
        "Probability": [prob_away, prob_draw, prob_home]
    })
    st.bar_chart(chart_data.set_index("Result"))