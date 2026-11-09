import os
import requests
import pandas as pd
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
ITUNES_URL = "https://itunes.apple.com/search"

st.set_page_config(
    page_title="RecomSphere AI | Discover",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- DESIGN ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
.stApp {
    background: #090b10;
    color: #f4f4f6;
}
[data-testid="stHeader"] {
    background: rgba(9,11,16,0.95);
}
.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}
section[data-testid="stSidebar"] {
    background: #10131b;
    border-right: 1px solid #252936;
}
.hero {
    background: linear-gradient(115deg, #22203e 0%, #171c31 48%, #102c30 100%);
    border: 1px solid #343750;
    border-radius: 24px;
    padding: 38px 36px;
    margin-bottom: 28px;
}
.hero-kicker {
    color: #9ee6cf;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
}
.hero h1 {
    color: white;
    font-size: clamp(32px, 5vw, 54px);
    font-weight: 800;
    line-height: 1.1;
    margin: 14px 0;
}
.hero p {
    color: #c1c7d5;
    font-size: 16px;
    max-width: 600px;
}
.section-title {
    font-size: 23px;
    font-weight: 700;
    margin: 28px 0 5px;
}
.section-subtitle {
    color: #9299aa;
    font-size: 13px;
    margin-bottom: 18px;
}
.track-card {
    background: #151923;
    border: 1px solid #292f3d;
    border-radius: 15px;
    padding: 12px;
    height: 100%;
    transition: border-color .2s ease;
}
.track-card:hover {
    border-color: #58d9b1;
}
.track-title {
    font-size: 14px;
    font-weight: 700;
    color: #ffffff;
    margin-top: 12px;
    min-height: 35px;
}
.track-artist {
    color: #a5adbd;
    font-size: 12px;
    min-height: 30px;
}
.track-score {
    color: #66dfb7;
    font-size: 12px;
    font-weight: 700;
    margin-top: 9px;
}
.track-rank {
    color: #66dfb7;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
}
div[data-testid="stMetric"] {
    background: #151923;
    border: 1px solid #292f3d;
    padding: 15px;
    border-radius: 13px;
}
.stButton > button {
    border-radius: 10px;
    min-height: 43px;
    font-weight: 700;
}
div[data-testid="stSelectbox"] > div,
div[data-testid="stSelectSlider"] {
    border-radius: 10px;
}
.footer {
    border-top: 1px solid #292f3d;
    margin-top: 50px;
    padding-top: 20px;
    color: #777f90;
    font-size: 12px;
}
</style>
""", unsafe_allow_html=True)


# ---------- API ----------
def api_get(endpoint, params=None):
    try:
        response = requests.get(
            f"{API_URL}{endpoint}",
            params=params,
            timeout=45
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        st.error(f"Recommendation API error: {exc}")
        return None


# ---------- ALBUM ARTWORK ----------
@st.cache_data(ttl=86400, show_spinner=False)
def get_album_art(track_name, artist_name):
    if not track_name or not artist_name:
        return None

    try:
        response = requests.get(
            ITUNES_URL,
            params={
                "term": f"{track_name} {artist_name}",
                "entity": "song",
                "limit": 8
            },
            timeout=12
        )
        response.raise_for_status()
        results = response.json().get("results", [])

        # Prefer an exact track + artist match.
        for item in results:
            if (
                item.get("trackName", "").casefold() == track_name.casefold()
                and item.get("artistName", "").casefold() == artist_name.casefold()
                and item.get("artworkUrl100")
            ):
                return item["artworkUrl100"].replace("100x100", "600x600")

        # Otherwise use the first search result with artwork.
        for item in results:
            if item.get("artworkUrl100"):
                return item["artworkUrl100"].replace("100x100", "600x600")

    except (requests.RequestException, ValueError):
        pass

    return None


# ---------- SIDEBAR ----------
st.sidebar.markdown("# 🎧 RecomSphere")
st.sidebar.caption("YOUR PERSONAL MUSIC DISCOVERY ENGINE")
st.sidebar.divider()

health = api_get("/health")

if health and health.get("model_loaded"):
    st.sidebar.success("Recommendation engine online")
    st.sidebar.metric("Listeners", f'{health["num_users"]:,}')
    st.sidebar.metric("Track catalogue", f'{health["num_tracks"]:,}')
else:
    st.sidebar.error("API unavailable")

st.sidebar.divider()

page = st.sidebar.radio(
    "EXPLORE",
    ["Discover Music", "Model Performance", "About"]
)

st.sidebar.divider()
st.sidebar.caption("Powered by User-Based Collaborative Filtering")


# ---------- DISCOVER ----------
if page == "Discover Music":

    st.markdown("""
    <div class="hero">
        <div class="hero-kicker">Music discovery, reimagined</div>
        <h1>Find music that<br>feels like you.</h1>
        <p>Explore personalised tracks inspired by your listening
        behaviour and people who share your musical taste.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Your listening profile</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Choose a Last.fm listener to discover their next favourite tracks.</div>',
        unsafe_allow_html=True
    )

    users_data = api_get("/users", params={"limit": 971})

    if users_data:
        users = users_data["users"]

        col1, col2 = st.columns([3, 1])

        with col1:
            selected_user = st.selectbox(
                "SELECT LISTENER",
                users,
                index=0
            )

        with col2:
            top_k = st.select_slider(
                "NUMBER OF TRACKS",
                options=[5, 10, 15, 20, 30, 50],
                value=10
            )

        if st.button(
            "Discover my music  →",
            type="primary",
            use_container_width=True
        ):
            with st.spinner("Finding music that matches this listener..."):
                result = api_get(
                    f"/recommend/{selected_user}",
                    params={"top_k": top_k}
                )

            if result:
                st.session_state["recommendations"] = result
                st.session_state["selected_user"] = selected_user

        result = st.session_state.get("recommendations")

        if result:
            recommendations = result["recommendations"]

            st.markdown("---")
            st.markdown(
                f'<div class="section-title">Made for {result["user_id"]}</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                '<div class="section-subtitle">A selection ranked using similar listeners.</div>',
                unsafe_allow_html=True
            )

            a, b, c = st.columns(3)
            a.metric("Tracks discovered", len(recommendations))
            b.metric("Similar listeners", "50")
            c.metric("Recommendation engine", "User-based CF")

            st.markdown(
                '<div class="section-title">Recommended for you</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                '<div class="section-subtitle">Your next listening session starts here.</div>',
                unsafe_allow_html=True
            )

            # Five-column responsive-looking artwork grid.
            for start in range(0, len(recommendations), 5):
                batch = recommendations[start:start + 5]
                columns = st.columns(5)

                for col, track in zip(columns, batch):
                    with col:
                        artwork = get_album_art(
                            track.get("track_name", ""),
                            track.get("artist_name", "")
                        )

                        with st.container(border=True):
                            st.markdown(
                                f'<div class="track-rank">TRACK {track["rank"]:02d}</div>',
                                unsafe_allow_html=True
                            )

                            if artwork:
                                st.image(artwork, use_container_width=True)
                            else:
                                st.image(
                                    "https://placehold.co/600x600/202638/ffffff?text=Music",
                                    use_container_width=True
                                )

                            st.markdown(
                                f'<div class="track-title">{track["track_name"]}</div>',
                                unsafe_allow_html=True
                            )
                            st.markdown(
                                f'<div class="track-artist">{track["artist_name"]}</div>',
                                unsafe_allow_html=True
                            )
                            st.markdown(
                                f'<div class="track-score">MATCH SCORE · {track["score"]:.5f}</div>',
                                unsafe_allow_html=True
                            )

            st.markdown("---")

            st.download_button(
                "↓  Download your recommendations",
                pd.DataFrame(recommendations).to_csv(index=False),
                file_name=f"{result['user_id']}_recommendations.csv",
                mime="text/csv",
                use_container_width=True
            )

        else:
            st.info("Choose a listener and click Discover my music to get started.")

    st.markdown("""
    <div class="footer">
        RECOMSPHERE AI · PERSONALIZED MUSIC DISCOVERY
        <br>Research and Development · Last.fm 1K Dataset
    </div>
    """, unsafe_allow_html=True)


# ---------- MODEL PERFORMANCE ----------
elif page == "Model Performance":

    st.markdown("## Model performance")
    st.caption("Experimental temporal holdout evaluation · 848 matched users")

    metrics = {
        "Model": ["Popularity", "Two-Tower", "User-Based CF"],
        "Precision@10": [0.01840, 0.00047, 0.04080],
        "Recall@10": [0.00057, 0.00002, 0.00414],
        "NDCG@10": [0.01895, 0.00042, 0.04268]
    }

    df = pd.DataFrame(metrics).set_index("Model")

    a, b, c = st.columns(3)
    a.metric("Precision@10", "0.04080", "2.22× vs popularity")
    b.metric("Recall@10", "0.00414", "7.26× vs popularity")
    c.metric("NDCG@10", "0.04268", "2.25× vs popularity")

    st.markdown("### Model comparison")
    st.dataframe(df, use_container_width=True)

    st.markdown("### Precision@10")
    st.bar_chart(df[["Precision@10"]])

    st.markdown("### NDCG@10")
    st.bar_chart(df[["NDCG@10"]])

    st.caption(
        "These are experimental results on the same temporal holdout "
        "evaluation population used during model development."
    )


# ---------- ABOUT ----------
elif page == "About":

    st.markdown("## About RecomSphere AI")

    st.markdown("""
    **RecomSphere AI** is a personalised music recommendation engine
    built using the Last.fm 1K listening history dataset.

    ### How it works

    1. Listening history is processed using PySpark.
    2. Implicit user-track interactions are constructed.
    3. Play counts are log-transformed and tracks are IDF-weighted.
    4. User similarity is calculated using cosine similarity.
    5. The 50 nearest users contribute to recommendation scores.
    6. Previously listened tracks are filtered out.
    7. The highest-ranked unseen tracks are returned.

    ### Technology stack

    - Python 3.11
    - PySpark
    - NumPy, Pandas and SciPy
    - Scikit-learn
    - PyTorch
    - FAISS
    - FastAPI and Uvicorn
    - Streamlit

    ### Dataset statistics

    | Metric | Value |
    |---|---:|
    | Users | 971 |
    | Track vocabulary | 1,419,059 |
    | Interactions | 4,303,070 |
    | Similar users | 50 |

    ### Recommendation model

    User-Based Collaborative Filtering
    """)

st.markdown("""
<div class="footer">
    RECOMSPHERE AI · BUILT WITH PYTHON AND MACHINE LEARNING
</div>
""", unsafe_allow_html=True)
