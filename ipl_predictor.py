import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="IPL-Zone", layout="wide")

# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv('matches_2008-2024.csv')
    df = df[df['result'] != 'no result']
    df = df[df['winner'].notna()]
    df['team1_win'] = (df['team1'] == df['winner']).astype(int)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    return df

df = load_data()

# Global LabelEncoder
team_le = LabelEncoder()

# Feature engineering
def prepare_features(df):
    df['team1_encoded'] = team_le.fit_transform(df['team1'])
    df['team2_encoded'] = team_le.transform(df['team2'])
    features = ['team1_encoded', 'team2_encoded']
    X = df[features]
    y = df['team1_win']
    return X, y

X, y = prepare_features(df)

# Train model
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return model, accuracy

model, accuracy = train_model(X, y)

# Get recent form for a team
def get_recent_form(team, n=5):
    team_matches = df[(df['team1'] == team) | (df['team2'] == team)]
    recent_matches = team_matches.tail(n)
    results = []
    for _, row in recent_matches.iterrows():
        result = 'Won' if row['winner'] == team else ('No result' if row['winner'] == 'NA' else 'Lost')
        opponent = row['team2'] if row['team1'] == team else row['team1']
        results.append({
            'date': row['date'].strftime('%Y-%m-%d'),
            'opponent': opponent,
            'result': result,
            'venue': row['venue']
        })
    return pd.DataFrame(results)

# Streamlit app
st.title(' IPL-Zone')

# Sidebar
st.sidebar.header('Team Selection')
team1 = st.sidebar.selectbox('Select Team 1', sorted(df['team1'].unique()))
team2 = st.sidebar.selectbox('Select Team 2', sorted(df['team2'].unique()))

# Season filter
years = sorted(df['date'].dt.year.unique(), reverse=True)
selected_year = st.sidebar.selectbox("Select Year for Analysis", ["All"] + years)
if selected_year != "All":
    df = df[df['date'].dt.year == selected_year]

st.sidebar.header('Model Information')
st.sidebar.write(f'Model Accuracy: {accuracy:.2%}')

if st.sidebar.button('Predict Winner'):
    def safe_transform(le, label, default=-1):
        return le.transform([label])[0] if label in le.classes_ else default

    team1_encoded = safe_transform(team_le, team1)
    team2_encoded = safe_transform(team_le, team2)

    if -1 in [team1_encoded, team2_encoded]:
        st.error("Prediction failed: selected team not in training data.")
    else:
        input_data = [[team1_encoded, team2_encoded]]
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        winner = team1 if prediction == 1 else team2
        winner_prob = probability[1] if prediction == 1 else probability[0]

        st.subheader('Prediction Result')
        st.success(f' Predicted Winner: {winner} ({(winner_prob * 100):.1f}% chance)')

        st.subheader('Recent Form')
        col1, col2 = st.columns(2)

        with col1:
            st.write(f'**{team1} - Last 5 Matches**')
            form1 = get_recent_form(team1)
            st.dataframe(form1, hide_index=True)
            fig1, ax1 = plt.subplots()
            form1['result'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax1)
            ax1.set_ylabel('')
            st.pyplot(fig1)

        with col2:
            st.write(f'**{team2} - Last 5 Matches**')
            form2 = get_recent_form(team2)
            st.dataframe(form2, hide_index=True)
            fig2, ax2 = plt.subplots()
            form2['result'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax2)
            ax2.set_ylabel('')
            st.pyplot(fig2)

    # Head-to-Head
    st.subheader('Head-to-Head Record')
    h2h = df[((df['team1'] == team1) & (df['team2'] == team2)) | ((df['team1'] == team2) & (df['team2'] == team1))]
    if not h2h.empty:
        h2h = h2h.copy()  # avoid SettingWithCopyWarning
        h2h['winner_team1'] = h2h['winner'] == h2h['team1']
        h2h_summary = h2h.groupby('winner_team1').size()
        team1_wins = h2h_summary.get(True, 0)
        team2_wins = h2h_summary.get(False, 0)

        st.write(f'**{team1} wins:** {team1_wins}')
        st.write(f'**{team2} wins:** {team2_wins}')

        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.bar([team1, team2], [team1_wins, team2_wins], color=['#1f77b4', '#ff7f0e'])
            ax.set_ylabel('Number of Wins')
            ax.set_title('Head-to-Head Record')
            st.pyplot(fig)

        with col2:
            st.subheader('Win Margin Distribution')
            fig3, ax3 = plt.subplots(figsize=(4, 3))
            h2h['result_margin'].dropna().astype(int).hist(bins=10, ax=ax3, color='green')
            ax3.set_title("Margin of Victory")
            ax3.set_xlabel("Runs/Wickets")
            ax3.set_ylabel("Match Count")
            st.pyplot(fig3)

        st.write('**Recent Matches Between These Teams**')
        recent_h2h = h2h.sort_values('date', ascending=False).head(5)
        st.dataframe(recent_h2h[['date', 'venue', 'winner', 'result_margin', 'result']], hide_index=True)

    else:
        st.warning('No historical matches found between these two teams.')

    # Team summary
    st.subheader(' Team Summary Stats')
    total_matches = pd.concat([df['team1'], df['team2']]).value_counts()
    total_wins = df['winner'].value_counts()
    summary = pd.DataFrame({
        'Matches Played': total_matches,
        'Matches Won': total_wins
    }).fillna(0).astype(int)
    st.dataframe(summary.loc[[team1, team2]])
