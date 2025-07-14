# 🎓 Top 1000 World Universities Dashboard

An interactive Streamlit dashboard for analyzing the Top 1000 World University rankings data.

## 📊 Features

- **Interactive Filters**: Filter by country, rank range, and score range
- **Multiple Visualizations**: 
  - Rankings and scores analysis
  - Geographic distribution with interactive map
  - Performance metrics correlation analysis
  - Detailed institution analysis
  - Statistical insights
  - University comparison tools
- **Real-time Data**: All visualizations update based on your filter selections
- **Export Functionality**: Download filtered data as CSV

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Installation Steps

1. **Clone or download the project files**
   Make sure you have the following files in your project directory:
   - `FP_Kelompok.py` (main dashboard file)
   - `Top 1000 World University.csv` (data file)
   - `requirements.txt` (dependencies)

2. **Install required dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   **Note**: The dashboard uses `statsmodels` for trendline analysis. If you encounter any issues with this dependency, the dashboard will automatically fall back to basic scatter plots without trendlines.

3. **Run the dashboard**
   ```bash
   streamlit run FP_Kelompok.py
   ```

4. **Access the dashboard**
   The dashboard will open automatically in your default web browser at `http://localhost:8501`

## 📁 File Structure

```
FP/
├── FP_Kelompok.py              # Main Streamlit dashboard
├── Top 1000 World University.csv  # Dataset
├── requirements.txt            # Python dependencies
└── README.md                  # This file
```

## 🎯 Dashboard Sections

### 1. 🏆 Rankings & Scores
- Top 20 universities by score
- Score distribution histogram
- World rank vs score scatter plot

### 2. 🌍 Geographic Distribution
- Universities by country
- Average scores by country
- Interactive world map with university locations

### 3. 📈 Performance Metrics
- Correlation matrix of all performance metrics
- Radar charts for top universities
- Box plots for key metrics

### 4. 🏛️ Institution Analysis
- Search for specific institutions
- Top institutions by different criteria
- Detailed institution profiles

### 5. 📊 Statistical Insights
- Summary statistics
- Score percentiles
- Performance trends
- Quality metrics comparison

### 6. 🔍 Search & Compare
- Multi-university comparison
- Side-by-side metrics comparison
- Radar chart comparisons
- Data export functionality

### 7. 🤖 Score Prediction
- Linear regression model for score prediction
- Interactive feature selection
- Model performance metrics (R², RMSE, MAE)
- Feature importance analysis
- Individual and batch prediction capabilities
- Confidence intervals and ranking estimates

## 🎛️ Interactive Features

### Sidebar Filters
- **Country Selection**: Choose specific countries to analyze
- **Rank Range**: Filter universities by world rank range
- **Score Range**: Filter universities by score range

### Real-time Updates
All visualizations and metrics update automatically when you change the filters in the sidebar.

### Export Data
Download the filtered dataset as a CSV file for further analysis in other tools.

## 📊 Data Source

The dashboard uses the "Top 1000 World University" dataset which includes:
- World and national rankings
- Overall scores
- Performance metrics (Quality of Education, Alumni Employment, etc.)
- Geographic coordinates
- Institution details

## 🛠️ Technical Details

- **Framework**: Streamlit
- **Visualization**: Plotly, Folium
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, Linear Regression
- **Styling**: Custom CSS

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. **File Not Found**: Ensure the CSV file is in the same directory as the Python file

3. **Port Already in Use**: If port 8501 is busy, Streamlit will automatically use the next available port

4. **Memory Issues**: For large datasets, consider filtering data before loading

### Getting Help

If you encounter any issues:
1. Check that all files are in the correct directory
2. Verify Python version (3.8+)
3. Ensure all dependencies are installed
4. Check the console output for error messages

## 📈 Usage Tips

1. **Start with Filters**: Use the sidebar filters to focus on specific countries or rank ranges
2. **Explore Different Tabs**: Each tab offers different insights and analysis
3. **Use the Search**: Find specific institutions quickly using the search feature
4. **Compare Universities**: Use the comparison tab to analyze multiple institutions side-by-side
5. **Export Data**: Download filtered results for external analysis

## 🎓 Educational Use

This dashboard is perfect for:
- University research and analysis
- Educational data visualization projects
- Academic performance studies
- Geographic distribution analysis
- Statistical analysis learning

---

**Created with ❤️ using Streamlit** 