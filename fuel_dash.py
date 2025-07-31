import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Fuel Efficiency Dashboard",
    page_icon="⛽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .kpi-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .kpi-label {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.5rem;
    }
    .inefficiency-alert {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_process_data():
    """Load and preprocess the data with comprehensive error handling"""
    try:
        # Load the raw data
        fuel_df = pd.read_excel("fuel_dashboard2.XLSX")
        prod_df = pd.read_excel("Prd Jun 2025.xlsx")
        
        # Fuel data preprocessing
        fuel_df['Equipment No.'] = fuel_df['Equipment No.'].fillna(fuel_df['Order'])
        
        # Drop unnecessary columns
        fuel_df = fuel_df.drop([
            "Material", "Material description", "Document Header Text", 
            "Storage location", "Movement type", "Special Stock", 
            "Material Document", "Material Doc.Item", "Plant", 'Unit of Entry',
            "User Name", 'Order', "Allocation", 'Item', "Purchase order", 
            'Reference', 'Valuation Type', 'Material Group Desc', 
            "Vendor", "Time of Entry"
        ], axis=1, errors='ignore')
        
        # Rename columns
        fuel_df = fuel_df.rename(columns={
            "Name 1": "Plant",
            "Posting Date": "Date"
        })
        
        # Make quantities and amounts positive
        fuel_df["Quantity"] = fuel_df["Qty in unit of entry"].abs()
        fuel_df["Amount"] = fuel_df["Amt.in Loc.Cur."].abs()
        fuel_df = fuel_df.drop(columns=["Qty in unit of entry", "Amt.in Loc.Cur."], errors='ignore')
        
        # Group fuel data by Date and Plant
        grouped_df = fuel_df.groupby(["Date", "Plant"], as_index=False)[["Quantity", "Amount"]].sum()
        
        # Production data preprocessing
        prod_df = prod_df.drop([
            "Status", "Factory", "Production", "ERP ID", "Time", "Recipe", 
            "Recipe Comment", "Exp. Class", 'Doses', 'Truck Driver', 
            'Pump Driver', 'Site', 'Deliv.Note', 'Project', 
            'Recipe Specifications', "Username", "Driver Code", "Shift Time", 
            'Comments', "Allocation", 'Notes', 'Day Name', "Contractor"
        ], axis=1, errors='ignore')
        
        production_df = prod_df.rename(columns={"Quantity (m³)": "Production_Quantity"})
        production_df["Plant"] = production_df["Plant"].str.lower().str.strip()
        
        # Group production by Date + Plant + Customer
        prod_grouped = production_df.groupby(["Date", "Plant", "Customer"], as_index=False).agg({
            "Production_Quantity": "sum"
        })
        
        # Calculate customer production percentages
        prod_grouped["Total_Production_Plant_Date"] = prod_grouped.groupby(["Date", "Plant"])["Production_Quantity"].transform("sum")
        prod_grouped["Customer_Production_Percentage"] = (
            prod_grouped["Production_Quantity"] / prod_grouped["Total_Production_Plant_Date"]
        ) * 100
        prod_grouped = prod_grouped.drop(["Total_Production_Plant_Date"], axis=1)
        
        # Normalize plant names for merging
        prod_grouped["Plant"] = prod_grouped["Plant"].str.strip().str.lower()
        grouped_df["Plant"] = grouped_df["Plant"].str.strip().str.lower()
        
        # Merge production and fuel data
        merged_df = pd.merge(prod_grouped, grouped_df, on=["Date", "Plant"], how="left")
        
        # Allocate fuel values based on production percentages
        merged_df["Fuel_Quantity_Allocated"] = (
            merged_df["Customer_Production_Percentage"] / 100 * merged_df["Quantity"]
        ).round(2)
        merged_df["Fuel_Amount_Allocated"] = (
            merged_df["Customer_Production_Percentage"] / 100 * merged_df["Amount"]
        ).round(2)
        
        # Calculate fuel efficiency metric
        merged_df["Fuel_per_m3"] = np.where(
            merged_df["Production_Quantity"] > 0,
            merged_df["Fuel_Quantity_Allocated"] / merged_df["Production_Quantity"],
            0
        ).round(4)
        
        # Keep only final columns
        merged_df = merged_df.drop(["Quantity", "Amount"], axis=1, errors='ignore')
        
        # Convert Date column to datetime
        merged_df["Date"] = pd.to_datetime(merged_df["Date"])
        
        # Remove rows with missing critical data
        merged_df = merged_df.dropna(subset=['Production_Quantity', 'Fuel_Quantity_Allocated'])
        
        return merged_df
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

def create_kpi_card(title, value, format_type="number"):
    """Create a KPI card with proper formatting"""
    if format_type == "currency":
        formatted_value = f"E£{value:,.2f}"
    elif format_type == "percentage":
        formatted_value = f"{value:.2f}%"
    elif format_type == "decimal":
        formatted_value = f"{value:.4f}"
    else:
        formatted_value = f"{value:,.0f}"
    
    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-value">{formatted_value}</div>
        <div class="kpi-label">{title}</div>
    </div>
    """, unsafe_allow_html=True)

def create_filters(df):
    """Create sidebar filters"""
    st.sidebar.header("🔧 Dashboard Filters")
    
    # Date range filter
    if not df.empty:
        min_date = df['Date'].min().date()
        max_date = df['Date'].max().date()
        
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        # Plant filter
        plants = ['All'] + sorted(df['Plant'].unique().tolist())
        selected_plants = st.sidebar.multiselect(
            "Select Plant(s)",
            plants,
            default='All'
        )
        
        # Customer filter - dynamically filtered based on selected plants
        if 'All' in selected_plants or not selected_plants:
            # Show all customers if 'All' plants selected or no plants selected
            available_customers = sorted(df['Customer'].unique().tolist())
        else:
            # Show only customers associated with selected plants
            filtered_df_for_customers = df[df['Plant'].isin(selected_plants)]
            available_customers = sorted(filtered_df_for_customers['Customer'].unique().tolist())
        
        customers = ['All'] + available_customers
        selected_customers = st.sidebar.multiselect(
            "Select Customer(s)",
            customers,
            default='All',
            help="Customers shown are filtered based on selected plant(s)"
        )
        
        # Calculate data-driven default threshold
        # Use 75th percentile + 1.5 * IQR (outlier detection method) or mean + 2*std
        fuel_per_m3_values = df['Fuel_per_m3'].dropna()
        if len(fuel_per_m3_values) > 0:
            q75 = fuel_per_m3_values.quantile(0.75)
            q25 = fuel_per_m3_values.quantile(0.25)
            iqr = q75 - q25
            data_driven_threshold = round(q75 + 1.5 * iqr, 2)
            
            # Ensure threshold is reasonable (not too extreme)
            max_reasonable = fuel_per_m3_values.quantile(0.95)
            default_threshold = min(data_driven_threshold, max_reasonable)
            
            # Ensure minimum threshold of 0.1
            default_threshold = max(default_threshold, 0.1)
        else:
            default_threshold = 2.0
        
        # Efficiency threshold for highlighting inefficiencies
        fuel_efficiency_threshold = st.sidebar.number_input(
            "Fuel Efficiency Threshold (per m³)",
            min_value=0.0,
            max_value=float(fuel_per_m3_values.max() * 1.2) if len(fuel_per_m3_values) > 0 else 10.0,
            value=default_threshold,
            step=0.1,
            help=f"Data-driven default: {default_threshold:.2f} (75th percentile + 1.5×IQR). Values above this threshold will be highlighted as inefficient"
        )
        
        return date_range, selected_plants, selected_customers, fuel_efficiency_threshold
    
    return None, ['All'], ['All'], 2.0

def filter_data(df, date_range, selected_plants, selected_customers):
    """Apply filters to the dataframe"""
    if df.empty:
        return df
    
    # Date filter
    if len(date_range) == 2:
        start_date, end_date = date_range
        df = df[(df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)]
    
    # Plant filter
    if 'All' not in selected_plants and selected_plants:
        df = df[df['Plant'].isin(selected_plants)]
    
    # Customer filter
    if 'All' not in selected_customers and selected_customers:
        df = df[df['Customer'].isin(selected_customers)]
    
    return df

def create_line_chart(df):
    """Create fuel efficiency over time line chart"""
    if df.empty:
        return go.Figure()
    
    # Group by Date and Plant only, averaging fuel efficiency
    plant_efficiency = df.groupby(['Date', 'Plant'])['Fuel_per_m3'].mean().reset_index()
    
    fig = px.line(
        plant_efficiency,
        x='Date',
        y='Fuel_per_m3',
        color='Plant',
        title='Fuel Efficiency (per m³) Over Time by Plant',
        labels={'Fuel_per_m3': 'Fuel per m³', 'Date': 'Date'}
    )
    
    fig.update_layout(
        height=500,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def create_bar_chart(df):
    """Create production vs fuel bar chart"""
    if df.empty:
        return go.Figure()
    
    customer_summary = df.groupby('Customer').agg({
        'Production_Quantity': 'sum',
        'Fuel_per_m3': 'mean'
    }).reset_index()
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Total Production by Customer', 'Average Fuel Efficiency by Customer'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Production bar chart
    fig.add_trace(
        go.Bar(
            x=customer_summary['Customer'],
            y=customer_summary['Production_Quantity'],
            name='Production (m³)',
            marker_color='lightblue'
        ),
        row=1, col=1
    )
    
    # Fuel efficiency bar chart
    fig.add_trace(
        go.Bar(
            x=customer_summary['Customer'],
            y=customer_summary['Fuel_per_m3'],
            name='Fuel per m³',
            marker_color='orange'
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        height=500,
        showlegend=False,
        title_text="Production vs Fuel Efficiency Analysis"
    )
    
    return fig

def create_pie_chart(df):
    """Create fuel allocation pie chart"""
    if df.empty:
        return go.Figure()
    
    fuel_by_customer = df.groupby('Customer')['Fuel_Amount_Allocated'].sum().reset_index()
    
    fig = px.pie(
        fuel_by_customer,
        values='Fuel_Amount_Allocated',
        names='Customer',
        title='Fuel Amount Allocation by Customer'
    )
    
    fig.update_layout(height=500)
    return fig

def highlight_inefficiencies(df, threshold):
    """Identify and display inefficiencies"""
    if df.empty:
        return
    
    inefficient_records = df[df['Fuel_per_m3'] > threshold]
    
    if not inefficient_records.empty:
        st.markdown(f"""
        <div class="inefficiency-alert">
            <h4>⚠️ Efficiency Alert</h4>
            <p>Found {len(inefficient_records)} records with fuel consumption above {threshold} per m³</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show top inefficient records
        top_inefficient = inefficient_records.nlargest(5, 'Fuel_per_m3')[
            ['Date', 'Plant', 'Customer', 'Production_Quantity', 'Fuel_per_m3']
        ]
        
        st.dataframe(
            top_inefficient,
            use_container_width=True,
            hide_index=True
        )

def create_forecast_chart(df):
    """Create a simple forecast using linear trend (Prophet alternative)"""
    if df.empty or len(df) < 10:
        st.info("Not enough data for forecasting (minimum 10 records required)")
        return
    
    try:
        # Aggregate daily fuel consumption
        daily_fuel = df.groupby('Date')['Fuel_Quantity_Allocated'].sum().reset_index()
        daily_fuel = daily_fuel.sort_values('Date')
        
        # Simple linear trend forecast
        from sklearn.linear_model import LinearRegression
        import sklearn
        
        # Prepare data for forecasting
        daily_fuel['days'] = (daily_fuel['Date'] - daily_fuel['Date'].min()).dt.days
        X = daily_fuel[['days']].values
        y = daily_fuel['Fuel_Quantity_Allocated'].values
        
        # Fit model
        model = LinearRegression()
        model.fit(X, y)
        
        # Generate future dates
        future_days = 30
        last_day = daily_fuel['days'].max()
        future_x = np.arange(last_day + 1, last_day + future_days + 1).reshape(-1, 1)
        future_dates = [daily_fuel['Date'].max() + timedelta(days=i) for i in range(1, future_days + 1)]
        
        # Make predictions
        future_y = model.predict(future_x)
        
        # Create forecast chart
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=daily_fuel['Date'],
            y=daily_fuel['Fuel_Quantity_Allocated'],
            mode='lines+markers',
            name='Historical',
            line=dict(color='blue')
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=future_y,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title='Fuel Demand Forecast (Next 30 Days)',
            xaxis_title='Date',
            yaxis_title='Fuel Quantity',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except ImportError:
        st.info("Install scikit-learn for forecasting: pip install scikit-learn")
    except Exception as e:
        st.error(f"Forecasting error: {str(e)}")

def main():
    """Main dashboard function"""
    # Header
    st.markdown('<h1 class="main-header">⛽ Fuel Efficiency Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading and processing data..."):
        df = load_and_process_data()
    
    if df.empty:
        st.error("No data available. Please check your data files.")
        return
    
    # Create filters
    date_range, selected_plants, selected_customers, threshold = create_filters(df)
    
    # Apply filters
    filtered_df = filter_data(df, date_range, selected_plants, selected_customers)
    
    if filtered_df.empty:
        st.warning("No data matches the selected filters.")
        return
    
    # KPI Section
    st.header("📊 Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_production = filtered_df['Production_Quantity'].sum()
        create_kpi_card("Total Production (m³)", total_production)
    
    with col2:
        total_fuel_quantity = filtered_df['Fuel_Quantity_Allocated'].sum()
        create_kpi_card("Total Fuel Quantity", total_fuel_quantity)
    
    with col3:
        total_fuel_amount = filtered_df['Fuel_Amount_Allocated'].sum()
        create_kpi_card("Total Fuel Amount", total_fuel_amount, "currency")
    
    with col4:
        avg_fuel_per_m3 = filtered_df['Fuel_per_m3'].mean()
        create_kpi_card("Avg Fuel per m³", avg_fuel_per_m3, "decimal")
    
    # Charts Section
    st.header("📈 Data Visualizations")
    
    # Line chart
    st.subheader("Fuel Efficiency Trends")
    line_fig = create_line_chart(filtered_df)
    st.plotly_chart(line_fig, use_container_width=True)
    
    # Bar and Pie charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Production Analysis")
        bar_fig = create_bar_chart(filtered_df)
        st.plotly_chart(bar_fig, use_container_width=True)
    
    with col2:
        st.subheader("Fuel Allocation Distribution")
        pie_fig = create_pie_chart(filtered_df)
        st.plotly_chart(pie_fig, use_container_width=True)
    
    # Forecast Section
    st.header("🔮 Fuel Demand Forecast")
    create_forecast_chart(filtered_df)
    
    # Efficiency Analysis
    st.header("⚡ Efficiency Analysis")
    highlight_inefficiencies(filtered_df, threshold)
    
    # Data Table
    st.header("📋 Detailed Data")
    
    # Create PDF export function
    def create_pdf_report(df, kpis):
        """Create a PDF report with dashboard data"""
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch
            import io
            
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, 
                                  topMargin=72, bottomMargin=18)
            
            # Styles
            styles = getSampleStyleSheet()
            title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], 
                                       fontSize=16, spaceAfter=30, alignment=1)
            
            # Story elements
            story = []
            
            # Title
            story.append(Paragraph("Fuel Efficiency Dashboard Report", title_style))
            story.append(Spacer(1, 20))
            
            # KPIs Section
            story.append(Paragraph("Key Performance Indicators", styles['Heading2']))
            kpi_data = [
                ['Metric', 'Value'],
                ['Total Production (m³)', f"{kpis['total_production']:,.2f}"],
                ['Total Fuel Quantity', f"{kpis['total_fuel_quantity']:,.2f}"],
                ['Total Fuel Amount', f"${kpis['total_fuel_amount']:,.2f}"],
                ['Average Fuel per m³', f"{kpis['avg_fuel_per_m3']:.4f}"]
            ]
            
            kpi_table = Table(kpi_data, colWidths=[3*inch, 2*inch])
            kpi_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(kpi_table)
            story.append(Spacer(1, 30))
            
            # Data Table
            story.append(Paragraph("Detailed Data", styles['Heading2']))
            
            # Prepare data for table (limit to first 50 rows for PDF)
            display_df = df.head(50).copy()
            
            # Format the data
            display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
            display_df['Production_Quantity'] = display_df['Production_Quantity'].round(2)
            display_df['Customer_Production_Percentage'] = display_df['Customer_Production_Percentage'].round(2)
            display_df['Fuel_Quantity_Allocated'] = display_df['Fuel_Quantity_Allocated'].round(2)
            display_df['Fuel_Amount_Allocated'] = display_df['Fuel_Amount_Allocated'].round(2)
            display_df['Fuel_per_m3'] = display_df['Fuel_per_m3'].round(4)
            
            # Create table data
            table_data = [display_df.columns.tolist()] + display_df.values.tolist()
            
            # Create table with smaller font for more columns
            data_table = Table(table_data, repeatRows=1)
            data_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 8),
                ('FONTSIZE', (0, 1), (-1, -1), 7),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(data_table)
            
            if len(df) > 50:
                story.append(Spacer(1, 12))
                story.append(Paragraph(f"Note: Showing first 50 rows of {len(df)} total records", 
                                     styles['Normal']))
            
            # Footer
            story.append(Spacer(1, 30))
            story.append(Paragraph(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                                 styles['Normal']))
            
            # Build PDF
            doc.build(story)
            buffer.seek(0)
            return buffer
            
        except ImportError:
            st.error("ReportLab not installed. Install with: pip install reportlab")
            return None
        except Exception as e:
            st.error(f"Error creating PDF: {str(e)}")
            return None
    
    # Prepare KPIs for PDF
    kpis = {
        'total_production': filtered_df['Production_Quantity'].sum(),
        'total_fuel_quantity': filtered_df['Fuel_Quantity_Allocated'].sum(),
        'total_fuel_amount': filtered_df['Fuel_Amount_Allocated'].sum(),
        'avg_fuel_per_m3': filtered_df['Fuel_per_m3'].mean()
    }
    
    # PDF download button
    
    #pdf_buffer = create_pdf_report(filtered_df, kpis)
    #if pdf_buffer:
        #st.download_button(
          #  label="📄 Download Dashboard Report as PDF",
          #  data=pdf_buffer.getvalue(),
          #  file_name=f"fuel_efficiency_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
           # mime="application/pdf"
      #  )
    
    # Display table with formatting
    st.dataframe(
        filtered_df.style.format({
            'Production_Quantity': '{:.2f}',
            'Customer_Production_Percentage': '{:.2f}%',
            'Fuel_Quantity_Allocated': '{:.2f}',
            'Fuel_Amount_Allocated': 'E£{:.2f}',
            'Fuel_per_m3': '{:.4f}'
        }).highlight_max(subset=['Fuel_per_m3'], color='lightcoral'),
        use_container_width=True,
        hide_index=True
    )
    
    # Footer
    st.markdown("---")
    st.markdown("**Dashboard created with Streamlit** | Data updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

if __name__ == "__main__":
    main()
