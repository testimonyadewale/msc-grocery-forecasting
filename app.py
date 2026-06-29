from flask import Flask, render_template, request, session, redirect, url_for
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from models import run_models
import os
import io

app = Flask(__name__)
app.secret_key = 'msc_grocery_2024'

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/')
def home():
    return render_template('base.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    message = None
    error   = False
    preview = None
    shape   = None
    columns = None

    if request.method == 'POST':
        file = request.files.get('file')

        if not file or file.filename == '':
            message = 'No file selected. Please choose a CSV file.'
            error   = True

        elif not file.filename.endswith('.csv'):
            message = 'Wrong file type. Please upload a CSV file only.'
            error   = True

        else:
            try:
                df       = pd.read_csv(file)
                required = ['date', 'store', 'item', 'sales']
                missing  = [c for c in required if c not in df.columns]

                if missing:
                    message = f'Missing columns: {missing}.'
                    error   = True
                else:
                    filepath = os.path.join(UPLOAD_FOLDER, 'train.csv')
                    df.to_csv(filepath, index=False)

                    session['data_loaded'] = True
                    session['rows']        = len(df)
                    session['cols']        = len(df.columns)

                    message = f'File uploaded successfully! {len(df):,} rows loaded.'
                    preview = df.head().to_dict('records')
                    columns = list(df.columns)
                    shape   = df.shape

            except Exception as e:
                message = f'Error reading file: {str(e)}'
                error   = True

    return render_template('upload.html',
                           message=message,
                           error=error,
                           preview=preview,
                           columns=columns,
                           shape=shape)


@app.route('/dashboard')
def dashboard():
    filepath = os.path.join(UPLOAD_FOLDER, 'train.csv')

    if not os.path.exists(filepath):
        return redirect(url_for('upload'))

    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])

    rows     = len(df)
    stores   = df['store'].nunique()
    items    = df['item'].nunique()
    date_min = df['date'].min().strftime('%d %b %Y')
    date_max = df['date'].max().strftime('%d %b %Y')

    daily = df.groupby('date')['sales'].sum().reset_index()
    fig1  = px.line(daily, x='date', y='sales',
                    title='Total Daily Sales 2013-2017',
                    color_discrete_sequence=['#2563A8'])
    fig1.update_layout(xaxis_title='Date',
                       yaxis_title='Total Units Sold',
                       plot_bgcolor='white',
                       height=350)
    chart1 = pio.to_html(fig1, full_html=False)

    df['day_of_week'] = df['date'].dt.day_name()
    day_order = ['Monday','Tuesday','Wednesday',
                 'Thursday','Friday','Saturday','Sunday']
    dow = (df.groupby('day_of_week')['sales']
             .mean()
             .reindex(day_order)
             .reset_index())
    fig2 = px.bar(dow, x='day_of_week', y='sales',
                  title='Average Sales by Day of Week',
                  color_discrete_sequence=['#1A3A5C'])
    fig2.update_layout(xaxis_title='Day',
                       yaxis_title='Average Units Sold',
                       plot_bgcolor='white',
                       height=320)
    chart2 = pio.to_html(fig2, full_html=False)

    df['month'] = df['date'].dt.month
    month_names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',
                   5:'May',6:'Jun',7:'Jul',8:'Aug',
                   9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
    monthly = df.groupby('month')['sales'].mean().reset_index()
    monthly['month_name'] = monthly['month'].map(month_names)
    fig3 = px.bar(monthly, x='month_name', y='sales',
                  title='Average Sales by Month',
                  color_discrete_sequence=['#2563A8'])
    fig3.update_layout(xaxis_title='Month',
                       yaxis_title='Average Units Sold',
                       plot_bgcolor='white',
                       height=320)
    chart3 = pio.to_html(fig3, full_html=False)

    return render_template('dashboard.html',
                           rows=f'{rows:,}',
                           stores=stores,
                           items=items,
                           date_min=date_min,
                           date_max=date_max,
                           chart1=chart1,
                           chart2=chart2,
                           chart3=chart3)


@app.route('/forecast')
def forecast():
    filepath = os.path.join(UPLOAD_FOLDER, 'train.csv')

    if not os.path.exists(filepath):
        return render_template('forecast.html', ready=False)

    results_df, chart_data, simulation = run_models(filepath)

    session['results']    = results_df.to_json()
    session['simulation'] = simulation

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=chart_data['date'],
        y=chart_data['actual'],
        name='Actual Sales',
        line=dict(color="#00A116", width=2)))

    fig.add_trace(go.Scatter(
        x=chart_data['date'],
        y=chart_data['xgb_pred'],
        name='XGBoost Forecast',
        line=dict(color="#AD0000", dash='dash')))

    fig.add_trace(go.Scatter(
        x=chart_data['date'],
        y=chart_data['ma_pred'],
        name='Moving Average',
        line=dict(color="#002679", dash='dot')))

    fig.update_layout(
        title='Actual vs Predicted — Store 1, Item 1 (2017)',
        xaxis_title='Date',
        yaxis_title='Units Sold',
        plot_bgcolor='white',
        height=420,
        legend=dict(x=0, y=1))

    chart = pio.to_html(fig, full_html=False)

    return render_template('forecast.html',
                           ready=True,
                           chart=chart)


@app.route('/compare')
def compare():
    if 'results' not in session:
        return render_template('compare.html', ready=False)

    import io

    results_df = pd.read_json(io.StringIO(session['results']))
    results_df = results_df.rename(columns={'Improvement_%': 'improvement'})
    results_df = results_df.round(4)
    results    = results_df.to_dict('records')
    best_improvement = results_df['improvement'].max()

    charts = []
    for _, row in results_df.iterrows():
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=['MAE'],
            y=[row['MAE']],
            name='MAE',
            marker_color='#DC2626',
            text=[row['MAE']],
            textposition='outside',
            hovertemplate=f'MAE: {row["MAE"]}<extra></extra>'))

        fig.add_trace(go.Bar(
            x=['RMSE'],
            y=[row['RMSE']],
            name='RMSE',
            marker_color='#2563A8',
            text=[row['RMSE']],
            textposition='outside',
            hovertemplate=f'RMSE: {row["RMSE"]}<extra></extra>'))

        fig.add_trace(go.Bar(
            x=['R\u00b2'],
            y=[row['R2']],
            name='R\u00b2',
            marker_color='#14532D',
            text=[row['R2']],
            textposition='outside',
            hovertemplate=f'R\u00b2: {row["R2"]}<extra></extra>'))

        fig.update_layout(
            title=f'{row["Model"]}',
            xaxis_title='Metric',
            yaxis_title='Score',
            plot_bgcolor='white',
            height=360,
            showlegend=True,
            bargap=0.3,
            uniformtext_minsize=8,
            uniformtext_mode='hide')

        charts.append(pio.to_html(fig, full_html=False))

    return render_template('compare.html',
                           ready=True,
                           results=results,
                           best_improvement=best_improvement,
                           charts=charts)

if __name__ == '__main__':
    app.run(debug=True)