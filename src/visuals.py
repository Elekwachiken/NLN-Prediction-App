import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def empty_figure(title="No Data Available", theme='light'):
    """
    Generates an empty Plotly figure with a central message.
    The colors adapt based on the provided theme.
    """
    if theme == 'dark':
        paper_bg = '#0a1f14' # Dark green from custom.css body.dark-mode
        plot_bg = '#1f2d24' # Darker green from custom.css page-content-wrapper.dark-mode
        font_color = '#e2ffe9' # Crisp readable text for dark mode
    else:
        paper_bg = '#f5f7fa' # Light gray from custom.css body
        plot_bg = '#ffffff' # White from custom.css page-content-wrapper
        font_color = '#333' # Default text color

    return go.Figure().update_layout(
        title=dict(text=title, x=0.5, font=dict(color=font_color)),
        template='plotly_dark' if theme == 'dark' else 'plotly_white',
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font=dict(color=font_color),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[dict(text="No data to display", x=0.5, y=0.5, showarrow=False, font=dict(size=20, color=font_color))]
    )

def style_figure(fig, title, text_color, bg_color, plot_bg_color, grid_color, template):
    """Applies consistent styling to Plotly figures."""
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18, color=text_color)),
        paper_bgcolor=bg_color,
        plot_bgcolor=plot_bg_color,
        font=dict(color=text_color),
        xaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color),
        yaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color),
        template=template
    )
    return fig


def generate_kpi_dashboard(game_df, wallet_df, theme='light'):
    """
    Generates KPI figures and values from game and wallet data.
    Each chart generation is wrapped in a try-except block to prevent
    one chart's failure from affecting others.
    """
    # Define theme-based styling
    if theme == 'dark':
        text_color = '#f1f1f1'
        bg_color = '#0a1f14' # Dark green from custom.css body.dark-mode
        plot_bg_color = '#1f2d24' # Darker green from custom.css page-content-wrapper.dark-mode
        grid_color = '#555'
        template = 'plotly_dark'
    else:
        text_color = '#333'
        bg_color = '#f5f7fa' # Light gray from custom.css body
        plot_bg_color = '#ffffff' # White from custom.css page-content-wrapper
        grid_color = '#e0e0e0'
        template = 'plotly_white'

    kpis = {
        'total_players': "0",
        'total_games_played': "0",
        'total_wallet_amount': "₦0.00",
        'game_fig': empty_figure("Game Product Distribution (No Data)", theme),
        'wallet_fig': empty_figure("Wallet Transaction Amounts (No Data)", theme),
        'dau_chart': empty_figure("Daily Active Users Over Time (No Data)", theme),
        'wallet_txn_by_action_channel_chart': empty_figure("Wallet Transactions by Action & Channel (No Data)", theme),
        'top_games_net_revenue_chart': empty_figure("Top 10 Games by Net Revenue (No Data)", theme),
        'top_games_total_plays_chart': empty_figure("Top 10 Games by Total Plays (No Data)", theme),
        'stake_prize_over_time_chart': empty_figure("Daily Total Stake vs. Prize Over Time (No Data)", theme),
        'engagement_by_hour_channel_chart': empty_figure("Player Engagement by Hour of Day and Access Channel (No Data)", theme),
        'access_channel_distribution_chart': empty_figure("Access Channel Distribution (No Data)", theme)
    }

    # Compute KPIs
    try:
        kpis['total_players'] = str(game_df['player_id'].nunique()) if 'player_id' in game_df and not game_df.empty else "0"
    except Exception as e:
        print(f"Error calculating total_players: {e}")

    try:
        kpis['total_games_played'] = str(game_df['stake'].count()) if 'stake' in game_df and not game_df.empty else "0"
    except Exception as e:
        print(f"Error calculating total_games_played: {e}")

    try:
        total_wallet_amount = wallet_df['amount'].sum() if 'amount' in wallet_df and not wallet_df.empty else 0
        kpis['total_wallet_amount'] = f"₦{total_wallet_amount:,.2f}"
    except Exception as e:
        print(f"Error calculating total_wallet_amount: {e}")


    # Bar chart for game product distribution
    try:
        if 'game' in game_df.columns and not game_df.empty:
            game_counts = game_df['game'].value_counts().reset_index()
            game_counts.columns = ['Game Product', 'Count']
            fig = px.bar(game_counts, x='Game Product', y='Count', text_auto=True)
            kpis['game_fig'] = style_figure(fig, 'Game Product Distribution', text_color, bg_color, plot_bg_color, grid_color, template)
    except Exception as e:
        print(f"Error generating Game Product Distribution chart: {e}")
        kpis['game_fig'] = empty_figure(f"Game Product Distribution (Error: {e})", theme)

    # Histogram for wallet transaction amounts
    try:
        if 'amount' in wallet_df.columns and not wallet_df.empty:
            fig = px.histogram(wallet_df, x='amount', nbins=30)
            kpis['wallet_fig'] = style_figure(fig, 'Wallet Transaction Amounts', text_color, bg_color, plot_bg_color, grid_color, template)
    except Exception as e:
        print(f"Error generating Wallet Transaction Amounts chart: {e}")
        kpis['wallet_fig'] = empty_figure(f"Wallet Transaction Amounts (Error: {e})", theme)

    # 1. Daily Active Users (DAU) Over Time
    try:
        if 'timestamp' in game_df.columns and 'player_id' in game_df.columns and not game_df.empty:
            dau_df = game_df.groupby(game_df['timestamp'].dt.date)['player_id'].nunique().reset_index()
            dau_df.columns = ['Date', 'Active Players']
            fig = px.line(dau_df, x='Date', y='Active Players')
            kpis['dau_chart'] = style_figure(fig, 'Daily Active Users Over Time', text_color, bg_color, plot_bg_color, grid_color, template)
    except Exception as e:
        print(f"Error generating Daily Active Users Over Time chart: {e}")
        kpis['dau_chart'] = empty_figure(f"Daily Active Users Over Time (Error: {e})", theme)

    # 2. Wallet Transaction Amounts by Action and Channel
    try:
        if 'amount' in wallet_df.columns and 'action' in wallet_df.columns and 'channel' in wallet_df.columns and not wallet_df.empty:
            wallet_agg = wallet_df.groupby(['action', 'channel'])['amount'].sum().reset_index()
            fig = px.bar(wallet_agg, x='channel', y='amount', color='action',
                                                    barmode='group',
                                                    color_discrete_map={'deposit': '#28a745', 'withdrawal': '#dc3545'},
                                                    text_auto=True)
            kpis['wallet_txn_by_action_channel_chart'] = style_figure(fig, 'Wallet Transaction Amounts by Action and Channel', text_color, bg_color, plot_bg_color, grid_color, template)
    except Exception as e:
        print(f"Error generating Wallet Transactions by Action & Channel chart: {e}")
        kpis['wallet_txn_by_action_channel_chart'] = empty_figure(f"Wallet Transactions by Action & Channel (Error: {e})", theme)

    # 3. Top 10 Games: Net Revenue vs. Total Plays
    try:
        if 'game' in game_df.columns and 'stake' in game_df.columns and 'prize' in game_df.columns and not game_df.empty:
            game_summary_for_kpi = game_df.groupby('game').agg(
                total_stake=('stake', 'sum'),
                total_prize=('prize', 'sum'),
                num_plays=('stake', 'count')
            ).reset_index()
            game_summary_for_kpi['net_revenue'] = game_summary_for_kpi['total_stake'] - game_summary_for_kpi['total_prize']

            if not game_summary_for_kpi.empty:
                top_10_net_revenue = game_summary_for_kpi.nlargest(10, 'net_revenue')
                fig_net_revenue = px.bar(top_10_net_revenue, x='game', y='net_revenue',
                                                     color_discrete_sequence=['#198754'], text_auto=True)
                kpis['top_games_net_revenue_chart'] = style_figure(fig_net_revenue, 'Top 10 Games by Net Revenue', text_color, bg_color, plot_bg_color, grid_color, template)

                top_10_total_plays = game_summary_for_kpi.nlargest(10, 'num_plays')
                fig_total_plays = px.bar(top_10_total_plays, x='game', y='num_plays',
                                                    color_discrete_sequence=['#0d6efd'], text_auto=True)
                kpis['top_games_total_plays_chart'] = style_figure(fig_total_plays, 'Top 10 Games by Total Plays', text_color, bg_color, plot_bg_color, grid_color, template)
    except Exception as e:
        print(f"Error generating Top Games charts: {e}")
        kpis['top_games_net_revenue_chart'] = empty_figure(f"Top 10 Games by Net Revenue (Error: {e})", theme)
        kpis['top_games_total_plays_chart'] = empty_figure(f"Top 10 Games by Total Plays (Error: {e})", theme)


    # 4. Daily Total Stake vs. Prize Over Time
    try:
        if 'timestamp' in game_df.columns and 'stake' in game_df.columns and 'prize' in game_df.columns and not game_df.empty:
            daily_financials = game_df.groupby(game_df['timestamp'].dt.date).agg(
                Total_Stake=('stake', 'sum'),
                Total_Prize=('prize', 'sum')
            ).reset_index()
            daily_financials.columns = ['Date', 'Total Stake', 'Total Prize']
            fig = px.line(daily_financials, x='Date', y=['Total Stake', 'Total Prize'],
                                              color_discrete_map={'Total Stake': '#198754', 'Total Prize': '#ffc107'})
            kpis['stake_prize_over_time_chart'] = style_figure(fig, 'Daily Total Stake vs. Prize Over Time', text_color, bg_color, plot_bg_color, grid_color, template)
    except Exception as e:
        print(f"Error generating Daily Stake vs. Prize chart: {e}")
        kpis['stake_prize_over_time_chart'] = empty_figure(f"Daily Total Stake vs. Prize Over Time (Error: {e})", theme)

    # 5. Player Engagement by Hour of Day and Access Channel (HEATMAP)
    try:
        if 'timestamp' in game_df.columns and 'access_channel' in game_df.columns and not game_df.empty:
            engagement_df = game_df.copy()
            engagement_df['hour'] = engagement_df['timestamp'].dt.hour
            engagement_by_time_channel = engagement_df.groupby(['hour', 'access_channel']).size().unstack(fill_value=0)

            ordered_channels = ['Web', 'USSD']
            existing_channels = [col for col in ordered_channels if col in engagement_by_time_channel.columns]
            other_channels = [col for col in engagement_by_time_channel.columns if col not in existing_channels]
            final_column_order = existing_channels + sorted(other_channels)

            final_column_order = [col for col in final_column_order if col in engagement_by_time_channel.columns]
            engagement_by_time_channel = engagement_by_time_channel[final_column_order]

            heatmap_data = [
                go.Heatmap(
                    z=engagement_by_time_channel.values,
                    x=engagement_by_time_channel.columns,
                    y=engagement_by_time_channel.index,
                    colorscale='Viridis',
                    colorbar=dict(title='Play Count'),
                    hovertemplate='<b>Hour:</b> %{y}<br><b>Channel:</b> %{x}<br><b>Plays:</b> %{z}<extra></extra>',
                    text=engagement_by_time_channel.values,
                    texttemplate="%{text}",
                    textfont={"size":10, "color": "white"}
                )
            ]
            fig = go.Figure(data=heatmap_data)
            fig = style_figure(fig, 'Player Engagement by Hour of Day and Access Channel', text_color, bg_color, plot_bg_color, grid_color, template)
            fig.update_xaxes(tickangle=45, tickfont=dict(size=10))
            fig.update_yaxes(tickfont=dict(size=10))
            fig.update_layout(height=600, width=800)
            kpis['engagement_by_hour_channel_chart'] = fig
    except Exception as e:
        print(f"Error generating Player Engagement Heatmap: {e}")
        kpis['engagement_by_hour_channel_chart'] = empty_figure(f"Player Engagement by Hour of Day and Access Channel (Error: {e})", theme)

    # 6. Access Channel Distribution (Pie Chart)
    try:
        if 'access_channel' in game_df.columns and not game_df.empty:
            channel_counts = game_df['access_channel'].value_counts().reset_index()
            channel_counts.columns = ['Channel', 'Count']
            fig = px.pie(channel_counts, names='Channel', values='Count', hole=0.3)
            kpis['access_channel_distribution_chart'] = style_figure(fig, 'Access Channel Distribution', text_color, bg_color, plot_bg_color, grid_color, template)
    except Exception as e:
        print(f"Error generating Access Channel Distribution chart: {e}")
        kpis['access_channel_distribution_chart'] = empty_figure(f"Access Channel Distribution (Error: {e})", theme)

    return kpis


def generate_churn_visuals(df, model_obj, theme='light'):
    """
    Generates churn prediction visualization figures.
    Each chart generation is wrapped in a try-except block to prevent
    one chart's failure from affecting others.
    """
    visuals = {}

    # Define theme-based style
    if theme == 'dark':
        text_color = '#f1f1f1'
        bg_color = '#0a1f14' # Dark green from custom.css body.dark-mode
        plot_bg_color = '#1f2d24' # Darker green from custom.css page-content-wrapper.dark-mode
        grid_color = '#555'
        template = 'plotly_dark'
    else:
        text_color = '#333'
        bg_color = '#f5f7fa' # Light gray from custom.css body
        plot_bg_color = '#ffffff' # White from custom.css page-content-wrapper
        grid_color = '#e0e0e0'
        template = 'plotly_white'

    # Helper for styling figures
    def style_fig(fig, title):
        return style_figure(fig, title, text_color, bg_color, plot_bg_color, grid_color, template)

    # Initialize all visuals to empty figures to handle potential errors gracefully
    visuals = {
        'churn-distribution-chart': empty_figure("Churn Distribution (No Data)", theme),
        'feature-importance-chart': empty_figure("Feature Importance (No Data)", theme),
        'most-played-game-chart': empty_figure("Most Played Games by Churn (No Data)", theme),
        'stake-vs-prize-chart': empty_figure("Stake vs Prize (No Data)", theme),
        'days-since-last-play-churn-hist': empty_figure("Days Since Last Play vs Churn (No Data)", theme),
        'tenure-churn-boxplot': empty_figure("Player Tenure vs Churn (No Data)", theme),
        'net-revenue-churn-boxplot': empty_figure("Net Revenue vs Churn (No Data)", theme),
        'player-value-segment-churn-chart': empty_figure("Player Value Segmentation by Churn (No Data)", theme),
        'weekly-churn-rate-chart': empty_figure("Weekly Churn Rate (No Data)", theme),
        'reactivation-chart': empty_figure("Player Reactivation After Win (No Data)", theme),
        'rfm-segment-churn-chart': empty_figure("RFM Segment Churn Analysis (No Data)", theme)
    }

    if df.empty or 'prediction' not in df.columns:
        return visuals # Return all empty figures if data is missing or invalid

    # Churn distribution (using 'prediction' column)
    try:
        churn_counts = df['prediction'].value_counts().reset_index(name='count')
        churn_counts.columns = ['Prediction', 'Count']
        bar_fig = px.bar(churn_counts, x='Prediction', y='Count', color='Prediction',
                         color_discrete_map={0: '#28a745', 1: '#dc3545'}, text_auto=True)
        visuals['churn-distribution-chart'] = style_fig(bar_fig, 'Churn Distribution')
    except Exception as e:
        print(f"Error generating Churn Distribution chart: {e}")
        visuals['churn-distribution-chart'] = empty_figure(f"Churn Distribution (Error: {e})", theme)

    # Feature importance
    try:
        if model_obj and hasattr(model_obj, 'feature_importances_') and hasattr(model_obj, 'feature_names_in_'):
            feat_imp_df = pd.DataFrame({
                'Feature': model_obj.feature_names_in_,
                'Importance': model_obj.feature_importances_
            }).sort_values(by='Importance', ascending=True)
            feat_fig = px.bar(feat_imp_df, x='Importance', y='Feature', orientation='h',
                              color_discrete_sequence=['#0077b6'], text_auto=True)
            visuals['feature-importance-chart'] = style_fig(feat_fig, 'Feature Importance')
        else:
            visuals['feature-importance-chart'] = empty_figure("Feature Importance (Model Not Loaded or Invalid)", theme)
    except Exception as e:
        print(f"Error generating Feature Importance chart: {e}")
        visuals['feature-importance-chart'] = empty_figure(f"Feature Importance (Error: {e})", theme)

    # Most Played Games by churn
    try:
        if 'most_played_game' in df.columns:
            game_fig = px.histogram(df, x='most_played_game', color='prediction', barmode='group',
                                    color_discrete_map={0: '#28a745', 1: '#dc3545'}, text_auto=True)
            visuals['most-played-game-chart'] = style_fig(game_fig, 'Most Played Games by Churn')
        else:
            visuals['most-played-game-chart'] = empty_figure("Most Played Games by Churn (Missing Data)", theme)
    except Exception as e:
        print(f"Error generating Most Played Games by Churn chart: {e}")
        visuals['most-played-game-chart'] = empty_figure(f"Most Played Games by Churn (Error: {e})", theme)

    # Stake vs Prize
    try:
        if all(col in df.columns for col in ['total_stake', 'total_prize']):
            scatter_fig = px.scatter(df, x='total_stake', y='total_prize', color='prediction',
                                     color_discrete_map={0: '#28a745', 1: '#dc3545'},
                                     hover_data=['player_id'])
            visuals['stake-vs-prize-chart'] = style_fig(scatter_fig, 'Stake vs Prize')
        else:
            visuals['stake-vs-prize-chart'] = empty_figure("Stake vs Prize (Missing 'total_stake' or 'total_prize')", theme)
    except Exception as e:
        print(f"Error generating Stake vs Prize chart: {e}")
        visuals['stake-vs-prize-chart'] = empty_figure(f"Stake vs Prize (Error: {e})", theme)

    # 1. Days Since Last Play Distribution (Churned vs. Non-Churned)
    try:
        if 'days_since_last_play' in df.columns and not df.empty:
            days_since_last_play_churn_hist = px.histogram(df, x='days_since_last_play', color='prediction',
                                                           barmode='overlay', histnorm='percent',
                                                           title='Days Since Last Play Distribution by Churn',
                                                           color_discrete_map={0: '#198754', 1: '#dc3545'})
            visuals['days-since-last-play-churn-hist'] = style_fig(days_since_last_play_churn_hist, 'Days Since Last Play Distribution by Churn')
        else:
            visuals['days-since-last-play-churn-hist'] = empty_figure("Days Since Last Play vs Churn (No 'days_since_last_play' data)", theme)
    except Exception as e:
        print(f"Error generating Days Since Last Play vs Churn chart: {e}")
        visuals['days-since-last-play-churn-hist'] = empty_figure(f"Days Since Last Play vs Churn (Error: {e})", theme)

    # 2. Player Tenure Distribution (Churned vs. Non-Churned)
    try:
        if 'Tenure_Days' in df.columns and not df.empty:
            tenure_churn_boxplot = px.box(df, x='prediction', y='Tenure_Days',
                                          color='prediction', title='Player Tenure vs Churn',
                                          color_discrete_map={0: '#198754', 1: '#dc3545'})
            visuals['tenure-churn-boxplot'] = style_fig(tenure_churn_boxplot, 'Player Tenure vs Churn')
        else:
            visuals['tenure-churn-boxplot'] = empty_figure("Player Tenure vs Churn (No 'Tenure_Days' data)", theme)
    except Exception as e:
        print(f"Error generating Player Tenure vs Churn chart: {e}")
        visuals['tenure-churn-boxplot'] = empty_figure(f"Player Tenure vs Churn (Error: {e})", theme)

    # 3. Net Revenue Distribution (Churned vs. Non-Churned)
    try:
        if 'net_revenue' in df.columns and not df.empty:
            net_revenue_churn_boxplot = px.box(df, x='prediction', y='net_revenue',
                                               color='prediction', title='Net Revenue vs Churn',
                                               color_discrete_map={0: '#198754', 1: '#dc3545'})
            visuals['net-revenue-churn-boxplot'] = style_fig(net_revenue_churn_boxplot, 'Net Revenue vs Churn')
        else:
            visuals['net-revenue-churn-boxplot'] = empty_figure("Net Revenue vs Churn (No 'net_revenue' data)", theme)
    except Exception as e:
        print(f"Error generating Net Revenue vs Churn chart: {e}")
        visuals['net-revenue-churn-boxplot'] = empty_figure(f"Net Revenue vs Churn (Error: {e})", theme)

    # 4. Churn Distribution by Player Value Segmentation
    try:
        if 'net_revenue' in df.columns and 'total_prize' in df.columns and not df.empty:
            # df['value_segment'] = pd.qcut(df['net_revenue'], q=4, labels=['Low Value', 'Medium-Low Value', 'Medium-High Value', 'High Value'], duplicates='drop')
            try:
                _, bins = pd.qcut(df['net_revenue'], q=4, retbins=True, duplicates='drop')
                num_bins = len(bins) - 1
                if num_bins >= 2:
                    value_labels = ['Low', 'Medium', 'High', 'Very High'][:num_bins]
                    df['value_segment'] = pd.qcut(df['net_revenue'], q=num_bins, labels=value_labels)
                else:
                    df['value_segment'] = 'Medium'  # fallback if not enough bins
            except Exception:
                df['value_segment'] = 'Medium'  # fallback if qcut still fails

            value_segment_churn_chart = px.box(df, x='value_segment', y='net_revenue', color='prediction',
                                                              title='Churn Distribution by Player Value Segmentation',
                                                              color_discrete_map={0: '#198754', 1: '#dc3545'})
            visuals['player-value-segment-churn-chart'] = style_fig(value_segment_churn_chart, 'Churn Distribution by Player Value Segmentation')
        else:
            visuals['player-value-segment-churn-chart'] = empty_figure("Player Value Segmentation by Churn (Missing data)", theme)
    except Exception as e:
        print(f"Error generating Player Value Segmentation by Churn chart: {e}")
        visuals['player-value-segment-churn-chart'] = empty_figure(f"Player Value Segmentation by Churn (Error: {e})", theme)

    # 5. Weekly Churn Rate
    try:
        if 'last_play_date' in df.columns and 'prediction' in df.columns and not df.empty:
            df['last_play_date'] = pd.to_datetime(df['last_play_date'])
            df['week_of_last_play'] = df['last_play_date'].dt.to_period('W').astype(str)

            weekly_churn_counts_raw = df.groupby('week_of_last_play')['prediction'].value_counts().unstack(fill_value=0)
            weekly_churn_counts = pd.DataFrame(weekly_churn_counts_raw).reindex(columns=[0, 1], fill_value=0)
            weekly_churn_counts.columns = ['Not Churned', 'Churned']

            weekly_churn_counts['Total Players'] = weekly_churn_counts['Not Churned'] + weekly_churn_counts['Churned']
            weekly_churn_counts['Churn Rate (%)'] = (weekly_churn_counts['Churned'] / weekly_churn_counts['Total Players'].replace(0, 1)) * 100

            weekly_churn_rate_chart = px.line(weekly_churn_counts.reset_index(), x='week_of_last_play', y='Churn Rate (%)',
                                              title='Weekly Churn Rate (Predicted Churners)',
                                              color_discrete_sequence=['#dc3545'])
            visuals['weekly-churn-rate-chart'] = style_fig(weekly_churn_rate_chart, 'Weekly Churn Rate (Predicted Churners)')
        else:
            visuals['weekly-churn-rate-chart'] = empty_figure("Weekly Churn Rate (Missing data)", theme)
    except Exception as e:
        print(f"Error generating Weekly Churn Rate chart: {e}")
        visuals['weekly-churn-rate-chart'] = empty_figure(f"Weekly Churn Rate (Error: {e})", theme)

    # 6. Player Reactivation After Win
    try:
        if 'reactivated' in df.columns and not df.empty:
            reactivation_counts = df['reactivated'].value_counts().rename({True: 'Reactivated', False: 'Not Reactivated'}).reset_index()
            reactivation_counts.columns = ['Status', 'Count']
            reactivation_chart = px.bar(reactivation_counts, x='Status', y='Count',
                                        title='Player Reactivation After First Win',
                                        color='Status',
                                        color_discrete_map={'Reactivated': '#198754', 'Not Reactivated': '#6c757d'},
                                        text_auto=True)
            visuals['reactivation-chart'] = style_fig(reactivation_chart, 'Player Reactivation After First Win')
        else:
            visuals['reactivation-chart'] = empty_figure("Player Reactivation After Win (No 'reactivated' data)", theme)
    except Exception as e:
        print(f"Error generating Player Reactivation After Win chart: {e}")
        visuals['reactivation-chart'] = empty_figure(f"Player Reactivation After Win (Error: {e})", theme)

    # 7: RFM Segment Churn Analysis
    try:
        if 'RFM_Segment' in df.columns and 'prediction' in df.columns and not df.empty:
            rfm_churn = df.groupby('RFM_Segment')['prediction'].value_counts(normalize=True).unstack(fill_value=0)
            rfm_churn = rfm_churn.reindex(columns=[0, 1], fill_value=0)
            rfm_churn['Churn Rate (%)'] = rfm_churn.get(1, 0) * 100

            segment_order = ['Champions', 'Loyal Customers', 'Potential Loyalists', 'New Customers', 'At Risk', 'Hibernating', 'Others']
            rfm_churn_display = rfm_churn.reindex(segment_order, fill_value=0).reset_index()

            rfm_segment_churn_chart = px.bar(rfm_churn_display, x='RFM_Segment', y='Churn Rate (%)',
                                             title='Churn Rate by RFM Segment',
                                             color='Churn Rate (%)',
                                             color_continuous_scale=px.colors.sequential.Reds,
                                             text_auto=True)
            visuals['rfm-segment-churn-chart'] = style_fig(rfm_segment_churn_chart, 'Churn Rate by RFM Segment')
        else:
            visuals['rfm-segment-churn-chart'] = empty_figure("RFM Segment Churn Analysis (Missing data)", theme)
    except Exception as e:
        print(f"Error generating RFM Segment Churn Analysis chart: {e}")
        visuals['rfm-segment-churn-chart'] = empty_figure(f"RFM Segment Churn Analysis (Error: {e})", theme)

    return visuals