import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import folium
from streamlit_folium import st_folium
import seaborn as sns
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Top 1000 World Universities Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('Top 1000 World University.csv', sep=';')
    # Clean data
    df['Score'] = pd.to_numeric(df['Score'], errors='coerce')
    df['World Rank'] = pd.to_numeric(df['World Rank'], errors='coerce')
    df['National Rank'] = pd.to_numeric(df['National Rank'], errors='coerce')
    df['Quality of Education'] = pd.to_numeric(df['Quality of Education'], errors='coerce')
    df['Alumni Employment'] = pd.to_numeric(df['Alumni Employment'], errors='coerce')
    df['Quality of Faculty'] = pd.to_numeric(df['Quality of Faculty'], errors='coerce')
    df['Research Output'] = pd.to_numeric(df['Research Output'], errors='coerce')
    df['Quality Publications'] = pd.to_numeric(df['Quality Publications'], errors='coerce')
    df['Influence'] = pd.to_numeric(df['Influence'], errors='coerce')
    df['Citations'] = pd.to_numeric(df['Citations'], errors='coerce')
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    return df

# Load the data
df = load_data()

# Header
st.markdown('<h1 class="main-header">🎓 Top 1000 World Universities Dashboard</h1>', unsafe_allow_html=True)

# Sidebar for filters
st.sidebar.header("📊 Dashboard Filters")

# Country filter
countries = sorted(df['Location'].unique())
selected_countries = st.sidebar.multiselect(
    "Select Countries:",
    countries,
    default=countries[:10]  # Default to first 10 countries
)

# Rank range filter
min_rank, max_rank = st.sidebar.slider(
    "World Rank Range:",
    min_value=int(df['World Rank'].min()),
    max_value=int(df['World Rank'].max()),
    value=(1, 100)
)

# Score range filter
min_score, max_score = st.sidebar.slider(
    "Score Range:",
    min_value=float(df['Score'].min()),
    max_value=float(df['Score'].max()),
    value=(float(df['Score'].min()), float(df['Score'].max()))
)

# Apply filters
filtered_df = df[
    (df['Location'].isin(selected_countries)) &
    (df['World Rank'] >= min_rank) &
    (df['World Rank'] <= max_rank) &
    (df['Score'] >= min_score) &
    (df['Score'] <= max_score)
]

# Main content
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Universities", len(filtered_df))
    
with col2:
    st.metric("Countries Represented", filtered_df['Location'].nunique())
    
with col3:
    st.metric("Average Score", f"{filtered_df['Score'].mean():.1f}")
    
with col4:
    st.metric("Top Ranked", filtered_df['World Rank'].min())

# Tabs for different visualizations
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏆 Rankings & Scores", 
    "🌍 Geographic Distribution", 
    "📈 Performance Metrics",
    "🏛️ Institution Analysis",
    "📊 Statistical Insights",
    "🔍 Search & Compare"
])

# Tab 1: Rankings & Scores
with tab1:
    st.header("🏆 University Rankings & Scores Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top 20 Universities by Score
        top_20 = filtered_df.nlargest(20, 'Score')
        fig_top20 = px.bar(
            top_20, 
            x='Score', 
            y='Institution',
            orientation='h',
            title="Top 20 Universities by Score",
            color='Score',
            color_continuous_scale='viridis'
        )
        fig_top20.update_layout(height=600)
        st.plotly_chart(fig_top20, use_container_width=True)
    
    with col2:
        # Score Distribution
        fig_dist = px.histogram(
            filtered_df, 
            x='Score', 
            nbins=30,
            title="Score Distribution",
            color_discrete_sequence=['#1f77b4']
        )
        fig_dist.update_layout(height=600)
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # World Rank vs Score Scatter Plot
    fig_scatter = px.scatter(
        filtered_df,
        x='World Rank',
        y='Score',
        color='Location',
        size='Quality of Faculty',
        hover_data=['Institution', 'National Rank'],
        title="World Rank vs Score (Size = Quality of Faculty)",
        size_max=20
    )
    fig_scatter.update_layout(height=500)
    st.plotly_chart(fig_scatter, use_container_width=True)

# Tab 2: Geographic Distribution
with tab2:
    st.header("🌍 Geographic Distribution of Universities")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Universities by Country
        country_counts = filtered_df['Location'].value_counts().head(15)
        fig_country = px.bar(
            x=country_counts.values,
            y=country_counts.index,
            orientation='h',
            title="Number of Universities by Country (Top 15)",
            color=country_counts.values,
            color_continuous_scale='plasma'
        )
        fig_country.update_layout(height=500)
        st.plotly_chart(fig_country, use_container_width=True)
    
    with col2:
        # Average Score by Country
        avg_score_by_country = filtered_df.groupby('Location')['Score'].mean().sort_values(ascending=False).head(15)
        fig_avg_score = px.bar(
            x=avg_score_by_country.values,
            y=avg_score_by_country.index,
            orientation='h',
            title="Average Score by Country (Top 15)",
            color=avg_score_by_country.values,
            color_continuous_scale='viridis'
        )
        fig_avg_score.update_layout(height=500)
        st.plotly_chart(fig_avg_score, use_container_width=True)
    
    # Interactive Map
    st.subheader("🗺️ Interactive World Map")
    
    # Create map
    map_df = filtered_df.dropna(subset=['Latitude', 'Longitude'])
    
    if not map_df.empty:
        m = folium.Map(location=[20, 0], zoom_start=2)
        
        for idx, row in map_df.iterrows():
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=5 + (row['Score'] / 10),
                popup=f"<b>{row['Institution']}</b><br>Rank: {row['World Rank']}<br>Score: {row['Score']:.1f}<br>Country: {row['Location']}",
                color='red' if row['World Rank'] <= 10 else 'blue',
                fill=True
            ).add_to(m)
        
        st_folium(m, width=800, height=400)
    else:
        st.warning("No geographic data available for the selected filters.")

# Tab 3: Performance Metrics
with tab3:
    st.header("📈 Performance Metrics Analysis")
    
    # Correlation heatmap
    st.subheader("📊 Correlation Matrix of Performance Metrics")
    
    metrics_cols = ['Quality of Education', 'Alumni Employment', 'Quality of Faculty', 
                   'Research Output', 'Quality Publications', 'Influence', 'Citations', 'Score']
    
    correlation_matrix = filtered_df[metrics_cols].corr()
    
    fig_heatmap = px.imshow(
        correlation_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu',
        title="Correlation Matrix of Performance Metrics"
    )
    fig_heatmap.update_layout(height=600)
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Performance metrics comparison
    col1, col2 = st.columns(2)
    
    with col1:
        # Radar chart for top university
        if not filtered_df.empty:
            top_university = filtered_df.loc[filtered_df['World Rank'].idxmin()]
            
            categories = ['Quality of Education', 'Alumni Employment', 'Quality of Faculty', 
                         'Research Output', 'Quality Publications', 'Influence', 'Citations']
            
            values = [top_university[cat] for cat in categories]
            
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=top_university['Institution']
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, max(values)])),
                showlegend=True,
                title=f"Performance Profile: {top_university['Institution']}"
            )
            st.plotly_chart(fig_radar, use_container_width=True)
    
    with col2:
        # Box plots for different metrics
        fig_box = px.box(
            filtered_df,
            y=['Quality of Education', 'Alumni Employment', 'Quality of Faculty'],
            title="Distribution of Key Performance Metrics"
        )
        fig_box.update_layout(height=400)
        st.plotly_chart(fig_box, use_container_width=True)

# Tab 4: Institution Analysis
with tab4:
    st.header("🏛️ Detailed Institution Analysis")
    
    # Search for specific institution
    search_term = st.text_input("🔍 Search for an institution:")
    
    if search_term:
        search_results = filtered_df[filtered_df['Institution'].str.contains(search_term, case=False, na=False)]
        
        if not search_results.empty:
            st.subheader(f"Search Results for '{search_term}'")
            
            for idx, row in search_results.iterrows():
                with st.expander(f"{row['Institution']} (Rank: {row['World Rank']})"):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("World Rank", row['World Rank'])
                        st.metric("National Rank", row['National Rank'])
                    
                    with col2:
                        st.metric("Score", f"{row['Score']:.1f}")
                        st.metric("Quality of Education", row['Quality of Education'])
                    
                    with col3:
                        st.metric("Alumni Employment", row['Alumni Employment'])
                        st.metric("Quality of Faculty", row['Quality of Faculty'])
                    
                    with col4:
                        st.metric("Research Output", row['Research Output'])
                        st.metric("Citations", row['Citations'])
        else:
            st.warning("No institutions found matching your search.")
    
    # Top institutions by different criteria
    st.subheader("🏆 Top Institutions by Different Criteria")
    
    criteria = st.selectbox(
        "Select ranking criteria:",
        ['Score', 'Quality of Education', 'Alumni Employment', 'Quality of Faculty', 
         'Research Output', 'Quality Publications', 'Influence', 'Citations']
    )
    
    top_by_criteria = filtered_df.nlargest(10, criteria)
    
    fig_criteria = px.bar(
        top_by_criteria,
        x=criteria,
        y='Institution',
        orientation='h',
        title=f"Top 10 Universities by {criteria}",
        color=criteria,
        color_continuous_scale='viridis'
    )
    fig_criteria.update_layout(height=500)
    st.plotly_chart(fig_criteria, use_container_width=True)

# Tab 5: Statistical Insights
with tab5:
    st.header("📊 Statistical Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Summary statistics
        st.subheader("📈 Summary Statistics")
        
        numeric_cols = ['Score', 'Quality of Education', 'Alumni Employment', 'Quality of Faculty', 
                       'Research Output', 'Quality Publications', 'Influence', 'Citations']
        
        summary_stats = filtered_df[numeric_cols].describe()
        st.dataframe(summary_stats, use_container_width=True)
        
        # Score percentiles
        st.subheader("📊 Score Distribution Percentiles")
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        score_percentiles = [filtered_df['Score'].quantile(p/100) for p in percentiles]
        
        percentile_df = pd.DataFrame({
            'Percentile': [f"{p}%" for p in percentiles],
            'Score': score_percentiles
        })
        st.dataframe(percentile_df, use_container_width=True)
    
    with col2:
        # Performance trends
        st.subheader("📈 Performance Trends")
        
        # Score vs Rank trend
        fig_trend = px.scatter(
            filtered_df,
            x='World Rank',
            y='Score',
            trendline="ols",
            title="Score vs World Rank Trend"
        )
        fig_trend.update_layout(height=400)
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Quality metrics comparison
        quality_metrics = ['Quality of Education', 'Quality of Faculty', 'Quality Publications']
        avg_quality = filtered_df[quality_metrics].mean()
        
        fig_quality = px.bar(
            x=quality_metrics,
            y=avg_quality.values,
            title="Average Quality Metrics",
            color=avg_quality.values,
            color_continuous_scale='viridis'
        )
        fig_quality.update_layout(height=300)
        st.plotly_chart(fig_quality, use_container_width=True)

# Tab 6: Search & Compare
with tab6:
    st.header("🔍 Search & Compare Universities")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 Search Universities")
        
        # Multi-select for universities
        universities = sorted(filtered_df['Institution'].unique())
        selected_unis = st.multiselect(
            "Select universities to compare:",
            universities,
            max_selections=5
        )
        
        if selected_unis:
            comparison_df = filtered_df[filtered_df['Institution'].isin(selected_unis)]
            
            # Display comparison table
            st.subheader("📊 Comparison Table")
            
            comparison_cols = ['Institution', 'World Rank', 'National Rank', 'Score', 
                             'Quality of Education', 'Alumni Employment', 'Quality of Faculty']
            
            st.dataframe(comparison_df[comparison_cols].set_index('Institution'), use_container_width=True)
    
    with col2:
        st.subheader("📈 Comparison Charts")
        
        if selected_unis and len(selected_unis) > 1:
            comparison_df = filtered_df[filtered_df['Institution'].isin(selected_unis)]
            
            # Radar chart comparison
            categories = ['Quality of Education', 'Alumni Employment', 'Quality of Faculty', 
                         'Research Output', 'Quality Publications', 'Influence', 'Citations']
            
            fig_radar_comp = go.Figure()
            
            for idx, row in comparison_df.iterrows():
                values = [row[cat] for cat in categories]
                fig_radar_comp.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name=row['Institution']
                ))
            
            fig_radar_comp.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1000])),
                showlegend=True,
                title="Performance Comparison"
            )
            st.plotly_chart(fig_radar_comp, use_container_width=True)
    
    # Download filtered data
    st.subheader("💾 Download Data")
    
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download filtered data as CSV",
        data=csv,
        file_name="filtered_universities.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🎓 Top 1000 World Universities Dashboard | Created with Streamlit</p>
        <p>Data source: Top 1000 World University Rankings</p>
    </div>
    """,
    unsafe_allow_html=True
)
