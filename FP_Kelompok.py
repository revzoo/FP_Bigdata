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
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(
    page_title="Dashboard Universitas Terbaik Dunia",
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
st.markdown('<h1 class="main-header">🎓 Dashboard 1000 Universitas Terbaik Dunia</h1>', unsafe_allow_html=True)

# Sidebar for filters
st.sidebar.header("📊 Filter Dashboard")

# Country filter
countries = sorted(df['Location'].unique())

# Add 'Pilih Semua Negara' button
if 'selected_countries' not in st.session_state:
    st.session_state.selected_countries = countries[:10]  # Default to first 10

if st.sidebar.button("Pilih Semua Negara"):
    st.session_state.selected_countries = countries

selected_countries = st.sidebar.multiselect(
    "Pilih Negara:",
    countries,
    default=st.session_state.selected_countries,
    key="selected_countries"
)

# Rank range filter
min_rank, max_rank = st.sidebar.slider(
    "Rentang Peringkat Dunia:",
    min_value=int(df['World Rank'].min()),
    max_value=int(df['World Rank'].max()),
    value=(1, 100)
)

# Score range filter
min_score, max_score = st.sidebar.slider(
    "Rentang Skor:",
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
    st.metric("Total Universitas", len(filtered_df))
    
with col2:
    st.metric("Negara Terwakili", filtered_df['Location'].nunique())
    
with col3:
    st.metric("Rata-rata Skor", f"{filtered_df['Score'].mean():.1f}")
    
with col4:
    st.metric("Peringkat Teratas", filtered_df['World Rank'].min())

# Tabs for different visualizations
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🏆 Peringkat & Skor", 
    "🌍 Distribusi Geografis", 
    "📈 Metrik Kinerja",
    "🏛️ Analisis Institusi",
    "📊 Statistik",
    "🔍 Cari & Bandingkan",
    "🤖 Prediksi Skor"
])

# Tab 1: Rankings & Scores
with tab1:
    st.header("🏆 Analisis Peringkat & Skor Universitas")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top 20 Universities by Score
        top_20 = filtered_df.nlargest(20, 'Score')
        fig_top20 = px.bar(
            top_20, 
            x='Score', 
            y='Institution',
            orientation='h',
            title="Top 20 Universitas Berdasarkan Skor",
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
            title="Distribusi Skor",
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
        title="Peringkat vs Skor (Ukuran = Kualitas Fasilitas)",
        size_max=20
    )
    fig_scatter.update_layout(height=500)
    st.plotly_chart(fig_scatter, use_container_width=True)

# Tab 2: Geographic Distribution
with tab2:
    st.header("🌍 Distribusi Geografis Universitas")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Universities by Country
        country_counts = filtered_df['Location'].value_counts().head(15)
        fig_country = px.bar(
            x=country_counts.values,
            y=country_counts.index,
            orientation='h',
            title="Jumlah Universitas Berdasarkan Negara (Top 15)",
            color=country_counts.values,
            color_continuous_scale='plasma'
        )
        fig_country.update_layout(height=500, xaxis_title='Jumlah Universitas', yaxis_title='Negara')
        st.plotly_chart(fig_country, use_container_width=True)
    
    with col2:
        # Average Score by Country
        avg_score_by_country = filtered_df.groupby('Location')['Score'].mean().sort_values(ascending=False).head(15)
        fig_avg_score = px.bar(
            x=avg_score_by_country.values,
            y=avg_score_by_country.index,
            orientation='h',
            title="Rata-rata Skor Berdasarkan Negara (Top 15)",
            color=avg_score_by_country.values,
            color_continuous_scale='viridis'
        )
        fig_avg_score.update_layout(height=500)
        st.plotly_chart(fig_avg_score, use_container_width=True)
    
    # Interactive Map
    st.subheader("🗺️ Peta Dunia Interaktif")
    
    # Create map
    map_df = filtered_df.dropna(subset=['Latitude', 'Longitude'])
    
    if not map_df.empty:
        m = folium.Map(location=[20, 0], zoom_start=2)
        
        for idx, row in map_df.iterrows():
            folium.CircleMarker(
                location=[row['Latitude'], row['Longitude']],
                radius=5 + (row['Score'] / 10),
                popup=f"<b>{row['Institution']}</b><br>Peringkat: {row['World Rank']}<br>Skor: {row['Score']:.1f}<br>Negara: {row['Location']}",
                color='red' if row['World Rank'] <= 10 else 'blue',
                fill=True
            ).add_to(m)
        
        st_folium(m, width=800, height=400)
    else:
        st.warning("Tidak ada data geografis tersedia untuk filter yang dipilih.")

# Tab 3: Performance Metrics
with tab3:
    st.header("📈 Analisis Metrik Kinerja")
    
    # Correlation heatmap
    st.subheader("📊 Matriks Korelasi Metrik Kinerja")
    
    metrics_cols = ['Quality of Education', 'Alumni Employment', 'Quality of Faculty', 
                   'Research Output', 'Quality Publications', 'Influence', 'Citations', 'Score']
    
    correlation_matrix = filtered_df[metrics_cols].corr()
    
    fig_heatmap = px.imshow(
        correlation_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu',
        title="Matriks Korelasi Metrik Kinerja"
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
                title=f"Profil Kinerja: {top_university['Institution']}"
            )
            st.plotly_chart(fig_radar, use_container_width=True)
    
    with col2:
        # Box plots for different metrics
        fig_box = px.box(
            filtered_df,
            y=['Quality of Education', 'Alumni Employment', 'Quality of Faculty'],
            title="Distribusi Metrik Kinerja Utama"
        )
        fig_box.update_layout(height=400)
        st.plotly_chart(fig_box, use_container_width=True)

# Tab 4: Institution Analysis
with tab4:
    st.header("🏛️ Analisis Institusi Detail")
    
    # Search for specific institution
    search_term = st.text_input("🔍 Cari institusi:")
    
    if search_term:
        search_results = filtered_df[filtered_df['Institution'].str.contains(search_term, case=False, na=False)]
        
        if not search_results.empty:
            st.subheader(f"Hasil Pencarian untuk '{search_term}'")
            
            for idx, row in search_results.iterrows():
                with st.expander(f"{row['Institution']} (Peringkat: {row['World Rank']})"):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Peringkat Dunia", row['World Rank'])
                        st.metric("Peringkat Nasional", row['National Rank'])
                    
                    with col2:
                        st.metric("Skor", f"{row['Score']:.1f}")
                        st.metric("Kualitas Pendidikan", row['Quality of Education'])
                    
                    with col3:
                        st.metric("Pekerjaan Lulusan", row['Alumni Employment'])
                        st.metric("Kualitas Fasilitas", row['Quality of Faculty'])
                    
                    with col4:
                        st.metric("Output Penelitian", row['Research Output'])
                        st.metric("Citations", row['Citations'])
        else:
            st.warning("Tidak ada institusi yang ditemukan sesuai pencarian Anda.")
    
    # Top institutions by different criteria
    st.subheader("🏆 Institusi Teratas Berdasarkan Kriteria Berbeda")
    
    criteria = st.selectbox(
        "Pilih kriteria peringkat:",
        ['Score', 'Quality of Education', 'Alumni Employment', 'Quality of Faculty', 
         'Research Output', 'Quality Publications', 'Influence', 'Citations']
    )
    
    top_by_criteria = filtered_df.nlargest(10, criteria)
    
    fig_criteria = px.bar(
        top_by_criteria,
        x=criteria,
        y='Institution',
        orientation='h',
        title=f"Top 10 Universitas Berdasarkan {criteria}",
        color=criteria,
        color_continuous_scale='viridis'
    )
    fig_criteria.update_layout(height=500)
    st.plotly_chart(fig_criteria, use_container_width=True)

# Tab 5: Statistical Insights
with tab5:
    st.header("📊 Statistik")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Summary statistics
        st.subheader("📈 Statistik Ringkasan")
        
        numeric_cols = ['Score', 'Quality of Education', 'Alumni Employment', 'Quality of Faculty', 
                       'Research Output', 'Quality Publications', 'Influence', 'Citations']
        
        summary_stats = filtered_df[numeric_cols].describe()
        st.dataframe(summary_stats, use_container_width=True)
        
        # Score percentiles
        st.subheader("📊 Distribusi Persentil Skor")
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        score_percentiles = [filtered_df['Score'].quantile(p/100) for p in percentiles]
        
        percentile_df = pd.DataFrame({
            'Persentil': [f"{p}%" for p in percentiles],
            'Skor': score_percentiles
        })
        st.dataframe(percentile_df, use_container_width=True)
    
    with col2:
        # Performance trends
        st.subheader("📈 Tren Kinerja")
        
        # Score vs Rank trend
        try:
            fig_trend = px.scatter(
                filtered_df,
                x='World Rank',
                y='Score',
                trendline="ols",
                title="Tren Skor vs Peringkat Dunia"
            )
        except ImportError:
            # Fallback if statsmodels is not available
            fig_trend = px.scatter(
                filtered_df,
                x='World Rank',
                y='Score',
                title="Tren Skor vs Peringkat Dunia (Tren tidak tersedia - statsmodels tidak tersedia)"
            )
        fig_trend.update_layout(height=400)
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Quality metrics comparison
        quality_metrics = ['Quality of Education', 'Quality of Faculty', 'Quality Publications']
        avg_quality = filtered_df[quality_metrics].mean()
        
        fig_quality = px.bar(
            x=quality_metrics,
            y=avg_quality.values,
            title="Rata-rata Metrik Kualitas",
            color=avg_quality.values,
            color_continuous_scale='viridis'
        )
        fig_quality.update_layout(height=300)
        st.plotly_chart(fig_quality, use_container_width=True)

# Tab 6: Search & Compare
with tab6:
    st.header("🔍 Cari & Bandingkan Universitas")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Cari Universitas")
        
        # Multi-select for universities
        universities = sorted(filtered_df['Institution'].unique())
        selected_unis = st.multiselect(
            "Pilih universitas untuk dibandingkan:",
            universities,
            max_selections=5
        )
        
        if selected_unis:
            comparison_df = filtered_df[filtered_df['Institution'].isin(selected_unis)]
            
            # Display comparison table
            st.subheader("📊 Tabel Perbandingan")
            
            comparison_cols = ['Institution', 'World Rank', 'National Rank', 'Score', 
                             'Quality of Education', 'Alumni Employment', 'Quality of Faculty']
            
            st.dataframe(comparison_df[comparison_cols].set_index('Institution'), use_container_width=True)
    
    with col2:
        st.subheader("📈 Grafik Perbandingan")
        
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
                title="Perbandingan Kinerja"
            )
            st.plotly_chart(fig_radar_comp, use_container_width=True)
    
    # Download filtered data
    st.subheader("💾 Unduh Data")
    
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Unduh data yang difilter sebagai CSV",
        data=csv,
        file_name="filtered_universities.csv",
        mime="text/csv"
    )

# Tab 7: Score Prediction
with tab7:
    st.header("Prediksi Skor Universitas dengan Regresi Linier")
    
    # Prepare data for modeling
    st.subheader("📊 Pengaturan Model")
    
    # Select features for prediction
    feature_options = ['Quality of Education', 'Alumni Employment', 'Quality of Faculty', 
                      'Research Output', 'Quality Publications', 'Influence', 'Citations']
    
    selected_features = st.multiselect(
        "Pilih fitur untuk digunakan dalam prediksi:",
        feature_options,
        default=feature_options[:5]  # Default to first 5 features
    )
    
    if len(selected_features) >= 2:
        # Prepare data
        model_df = filtered_df[selected_features + ['Score']].dropna()
        
        if len(model_df) > 10:  # Ensure we have enough data
            X = model_df[selected_features]
            y = model_df['Score']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)
            
            # Calculate metrics
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            # Display model performance
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("R² Skor Pelatihan", f"{train_r2:.3f}")
                st.metric("RMSE Pelatihan", f"{train_rmse:.2f}")
            
            with col2:
                st.metric("R² Skor Uji", f"{test_r2:.3f}")
                st.metric("RMSE Uji", f"{test_rmse:.2f}")
            
            with col3:
                st.metric("MAE Pelatihan", f"{train_mae:.2f}")
                st.metric("MAE Uji", f"{test_mae:.2f}")
            
            # Model coefficients
            st.subheader("📈 Koefisien Model")
            coefficients_df = pd.DataFrame({
                'Fitur': selected_features,
                'Koefisien': model.coef_
            }).sort_values('Koefisien', key=abs, ascending=False)
            
            fig_coeff = px.bar(
                coefficients_df,
                x='Koefisien',
                y='Fitur',
                orientation='h',
                title="Importansi Fitur (Ukuran Koefisien)",
                color='Koefisien',
                color_continuous_scale='RdBu'
            )
            fig_coeff.update_layout(height=400)
            st.plotly_chart(fig_coeff, use_container_width=True)
            
            # Prediction vs Actual
            st.subheader("🎯 Nilai Prediksi vs Nilai Aktual")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Training data
                fig_train = px.scatter(
                    x=y_train,
                    y=y_pred_train,
                    title="Data Pelatihan: Aktual vs Prediksi",
                    labels={'x': 'Skor Aktual', 'y': 'Skor Prediksi'}
                )
                fig_train.add_trace(go.Scatter(
                    x=[y_train.min(), y_train.max()],
                    y=[y_train.min(), y_train.max()],
                    mode='lines',
                    name='Prediksi Sempurna',
                    line=dict(dash='dash', color='red')
                ))
                fig_train.update_layout(height=400)
                st.plotly_chart(fig_train, use_container_width=True)
            
            with col2:
                # Test data
                fig_test = px.scatter(
                    x=y_test,
                    y=y_pred_test,
                    title="Data Uji: Aktual vs Prediksi",
                    labels={'x': 'Skor Aktual', 'y': 'Skor Prediksi'}
                )
                fig_test.add_trace(go.Scatter(
                    x=[y_test.min(), y_test.max()],
                    y=[y_test.min(), y_test.max()],
                    mode='lines',
                    name='Prediksi Sempurna',
                    line=dict(dash='dash', color='red')
                ))
                fig_test.update_layout(height=400)
                st.plotly_chart(fig_test, use_container_width=True)
            
            # Interactive Prediction
            st.subheader("🔮 Buat Prediksi Sendiri")
            st.write("Masukkan nilai untuk fitur yang dipilih untuk memprediksi skor universitas:")
            
            # Create input fields for features
            input_data = {}
            col1, col2 = st.columns(2)
            
            for i, feature in enumerate(selected_features):
                with col1 if i % 2 == 0 else col2:
                    # Get min and max values for the feature
                    min_val = float(filtered_df[feature].min())
                    max_val = float(filtered_df[feature].max())
                    mean_val = float(filtered_df[feature].mean())
                    
                    input_data[feature] = st.slider(
                        f"{feature}:",
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val,
                        step=(max_val - min_val) / 100
                    )
            
            # Auto-calculate prediction when sliders change
            input_df = pd.DataFrame([input_data])
            input_scaled = scaler.transform(input_df)
            predicted_score = model.predict(input_scaled)[0]
            
            # Display result
            st.success(f"🎯 **Skor Prediksi: {predicted_score:.2f}**")
            
            # Show confidence interval (simplified)
            confidence_range = test_rmse * 1.96  # 95% confidence interval
            st.info(f"📊 **Rentang Kepercayaan: {predicted_score - confidence_range:.2f} hingga {predicted_score + confidence_range:.2f}**")
            
            # Show what this score might mean
            if predicted_score >= 90:
                rank_category = "Universitas Top 10-20"
            elif predicted_score >= 80:
                rank_category = "Universitas Top 50-100"
            elif predicted_score >= 70:
                rank_category = "Universitas Top 200-500"
            else:
                rank_category = "Universitas Top 500+"
            
            st.info(f"🏆 **Kategori Peringkat Taksiran: {rank_category}**")
    
    else:
        st.warning("⚠️ Silakan pilih setidaknya 2 fitur untuk prediksi.")
    
    # Model explanation
    st.subheader("📚 Cara Kerja Model")
    st.markdown("""
    **Model Regresi Linier:**
    - Menggunakan metrik kinerja yang dipilih untuk memprediksi skor universitas
    - Melatih pada 80% data dan menguji pada 20%
    - Fitur distandarisasi (diskalakan) untuk kinerja lebih baik
    - Skor R² menunjukkan seberapa baik model menjelaskan variasi skor
    - RMSE dan MAE menunjukkan akurasi prediksi
    
    **Interpretasi:**
    - R² lebih tinggi = Model lebih baik
    - RMSE/MAE lebih rendah = Prediksi lebih akurat
    - Koefisien positif = Nilai fitur lebih tinggi meningkatkan skor prediksi
    - Koefisien negatif = Nilai fitur lebih tinggi menurunkan skor prediksi
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🎓 Dashboard 1000 Universitas Terbaik Dunia | Dibuat dengan Streamlit</p>
        <p>Sumber data: Peringkat 1000 Universitas Terbaik Dunia</p>
    </div>
    """,
    unsafe_allow_html=True
)
