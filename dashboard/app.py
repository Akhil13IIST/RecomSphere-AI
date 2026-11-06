
import requests
import pandas as pd
import streamlit as st

# --------------------------------------------------
# Configuration
# --------------------------------------------------

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="RecomSphere AI",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# Styling
# --------------------------------------------------

st.markdown("""
<style>
    .stApp {
        background-color: #0b1020;
        color: #f1f5f9;
    }

    [data-testid="stSidebar"] {
        background-color: #111827;
    }

    .hero {
        padding: 28px;
        border-radius: 16px;
        background: linear-gradient(120deg, #172554, #312e81, #581c87);
        margin-bottom: 25px;
    }

    .hero h1 {
        font-size: 36px;
        margin-bottom: 5px;
        color: white;
    }

    .hero p {
        color: #dbeafe;
        font-size: 16px;
    }

    .movie-card {
        background: #151e31;
        border: 1px solid #263449;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
        min-height: 145px;
    }

    .movie-card h4 {
        color: #f8fafc;
        margin-bottom: 8px;
        font-size: 16px;
    }

    .movie-card p {
        color: #a5b4c8;
        font-size: 13px;
    }

    .score {
        color: #a5b4fc;
        font-weight: bold;
    }

    div[data-testid="stMetric"] {
        background: #151e31;
        border: 1px solid #263449;
        padding: 15px;
        border-radius: 12px;
    }

    h2, h3 {
        color: #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# API helpers
# --------------------------------------------------

def api_get(endpoint, params=None):
    try:
        response = requests.get(
            f"{API_URL}{endpoint}",
            params=params,
            timeout=15
        )
        response.raise_for_status()
        return response.json(), None

    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to the API. Start FastAPI on port 8000."

    except requests.exceptions.Timeout:
        return None, "The API request timed out."

    except requests.exceptions.HTTPError as error:
        return None, f"API error: {error}"

    except requests.exceptions.RequestException as error:
        return None, str(error)


def display_movies(movies, score_label="Recommendation score"):
    if not movies:
        st.info("No movies found.")
        return

    for start in range(0, len(movies), 2):
        columns = st.columns(2)

        for offset, column in enumerate(columns):
            index = start + offset

            if index >= len(movies):
                break

            movie = movies[index]

            with column:
                st.markdown(
                    f"""
                    <div class="movie-card">
                        <h4>{movie.get("title", "Unknown movie")}</h4>
                        <p>{movie.get("genres", "Genres unavailable")}</p>
                        <p class="score">
                            {score_label}: {movie.get("score", movie.get("similarity", "N/A"))}
                        </p>
                        <p>Movie ID: {movie.get("movie_id", "N/A")}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("🎬 RecomSphere AI")
st.sidebar.caption("Hybrid Personalized Movie Recommendation Engine")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Recommendations",
        "Movie Discovery",
        "Similar Movies",
        "Model Performance"
    ]
)

st.sidebar.divider()

health, health_error = api_get("/health")

if health:
    st.sidebar.success("API Online")
    st.sidebar.metric("Registered users", health["users"])
    st.sidebar.metric("Movie catalogue", health["movies"])
else:
    st.sidebar.error("API Offline")
    st.sidebar.caption(health_error)


# --------------------------------------------------
# Home
# --------------------------------------------------

if page == "Home":

    st.markdown("""
    <div class="hero">
        <h1>RecomSphere AI</h1>
        <p>Your personalized movie discovery platform.</p>
        <p>Discover films using collaborative filtering,
        content-based intelligence and hybrid recommendations.</p>
    </div>
    """, unsafe_allow_html=True)

    if health:
        col1, col2, col3 = st.columns(3)

        col1.metric("Users", health["users"])
        col2.metric("Movies", health["movies"])
        col3.metric("Recommendation approaches", "3")

    st.subheader("Explore RecomSphere")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Personalized recommendations")
        st.write(
            "Get movie suggestions based on user preferences "
            "and collaborative patterns."
        )

    with col2:
        st.markdown("### Intelligent discovery")
        st.write(
            "Find similar movies using genre and title-based "
            "content representations."
        )

    st.info("Select a section from the sidebar to get started.")


# --------------------------------------------------
# Recommendations
# --------------------------------------------------

elif page == "Recommendations":

    st.title("Personalized Recommendations")
    st.write("Discover movies selected for a MovieLens user.")

    col1, col2, col3 = st.columns(3)

    with col1:
        user_id = st.number_input(
            "User ID",
            min_value=1,
            max_value=943,
            value=1,
            step=1
        )

    with col2:
        method = st.selectbox(
            "Recommendation method",
            ["collaborative", "content", "hybrid"]
        )

    with col3:
        limit = st.selectbox(
            "Number of movies",
            [5, 10, 15, 20, 30, 50],
            index=1
        )

    method_descriptions = {
        "collaborative": "Finds movies based on patterns in similar users' ratings.",
        "content": "Finds movies based on the genres and characteristics of liked films.",
        "hybrid": "Combines collaborative and content-based recommendation scores."
    }

    st.caption(method_descriptions[method])

    if st.button("Generate Recommendations", type="primary"):
        with st.spinner("Finding movies for you..."):

            data, error = api_get(
                f"/recommend/{user_id}",
                params={
                    "limit": limit,
                    "method": method
                }
            )

        if error:
            st.error(error)

        else:
            movies = data.get("recommendations", [])

            st.success(
                f"Generated {len(movies)} recommendations "
                f"for user {user_id} using {method} filtering."
            )

            display_movies(movies)


# --------------------------------------------------
# Movie Discovery
# --------------------------------------------------

elif page == "Movie Discovery":

    st.title("Movie Discovery")
    st.write("Search the MovieLens catalogue by movie title.")

    query = st.text_input(
        "Search movies",
        placeholder="Try Star Wars, Fargo, Toy Story..."
    )

    if st.button("Search Movies", type="primary"):

        if not query.strip():
            st.warning("Enter a movie title first.")

        else:
            data, error = api_get(
                "/movies/search",
                params={
                    "q": query.strip(),
                    "limit": 20
                }
            )

            if error:
                st.error(error)

            else:
                results = data.get("results", [])

                st.write(f"Found {len(results)} matching movies.")

                display_movies(
                    results,
                    score_label="Movie ID"
                )


# --------------------------------------------------
# Similar Movies
# --------------------------------------------------

elif page == "Similar Movies":

    st.title("Similar Movie Finder")
    st.write(
        "Find movies with similar genre and title characteristics."
    )

    movie_id = st.number_input(
        "Enter a Movie ID",
        min_value=1,
        max_value=1682,
        value=2,
        step=1
    )

    limit = st.slider(
        "Number of similar movies",
        min_value=5,
        max_value=50,
        value=10
    )

    if st.button("Find Similar Movies", type="primary"):

        data, error = api_get(
            f"/similar/{movie_id}",
            params={"limit": limit}
        )

        if error:
            st.error(error)

        else:
            st.subheader(
                f"Movies similar to {data.get('source_movie', 'Selected movie')}"
            )

            display_movies(
                data.get("similar_movies", []),
                score_label="Similarity"
            )


# --------------------------------------------------
# Model Performance
# --------------------------------------------------

elif page == "Model Performance":

    st.title("Model Performance")
    st.write("Offline evaluation using the MovieLens 100K dataset.")

    st.caption(
        "Metrics below use the same per-user chronological "
        "80/20 evaluation split and Precision@10, Recall@10, "
        "NDCG@10 and Hit Rate@10."
    )

    performance = pd.DataFrame({
        "Model": [
            "Popularity",
            "Collaborative Filtering",
            "Content-Based",
            "Hybrid"
        ],
        "Precision@10": [
            0.0527, 0.1244, 0.0216, 0.1236
        ],
        "Recall@10": [
            0.0479, 0.1425, 0.0210, 0.1372
        ],
        "NDCG@10": [
            0.0574, 0.1735, 0.0265, 0.1679
        ],
        "Hit Rate@10": [
            0.3164, 0.5943, 0.1698, 0.5976
        ]
    })

    st.subheader("Evaluation comparison")

    st.dataframe(
        performance.set_index("Model").style.format("{:.4f}"),
        use_container_width=True
    )

    st.subheader("Precision@10 comparison")

    st.bar_chart(
        performance.set_index("Model")[["Precision@10"]]
    )

    st.subheader("NDCG@10 comparison")

    st.bar_chart(
        performance.set_index("Model")[["NDCG@10"]]
    )

    st.subheader("Key observations")

    st.markdown("""
    - Collaborative filtering achieves the strongest Precision@10,
      Recall@10 and NDCG@10.
    - Hybrid filtering achieves the highest Hit Rate@10.
    - Content-based filtering provides an alternative recommendation
      strategy but performs less strongly on this evaluation.
    - The hybrid approach is competitive, but does not outperform
      collaborative filtering on every metric.
    """)

    st.info(
        "These are offline experimental results, not live production "
        "performance guarantees."
    )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "RecomSphere AI | MovieLens 100K | "
    "FastAPI + Streamlit + Scikit-learn"
)
