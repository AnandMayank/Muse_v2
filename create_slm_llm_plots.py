#!/usr/bin/env python3
"""
SLM vs LLM MUSE Comparison Visualization Script
==============================================

Creates comprehensive visualizations for the SLM vs LLM evaluation results comparing:
- Small Language Models (Phi-3-mini, Gemma-2B, Llama-3.2-1B)
- Large Language Models (GPT-4, GPT-3.5-turbo)

Key visualizations:
- Performance comparison across quality metrics
- Efficiency analysis (speed vs cost)
- Model ranking comparisons
- Trade-off analysis between quality and efficiency
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

# Set up plotting style
plt.style.use('default')
sns.set_palette("Set2")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

def load_slm_llm_results(file_path: str) -> dict:
    """Load SLM vs LLM evaluation results"""
    with open(file_path, 'r') as f:
        return json.load(f)

def create_slm_llm_comparison_dashboard(analysis: dict):
    """Create comprehensive SLM vs LLM comparison dashboard"""
    
    # Extract model data
    model_rankings = analysis["model_rankings"]["by_quality"]
    
    # Prepare data for plotting
    models_data = []
    for model_key, model_info in model_rankings:
        model_type = model_info["model_type"]
        model_name = model_info["model_name"]
        
        models_data.append({
            'Model': f"{model_name}",
            'Type': model_type,
            'User_Satisfaction': model_info["user_satisfaction_score"],
            'Response_Coherence': model_info["response_coherence"],
            'Tool_Accuracy': model_info["tool_selection_accuracy"],
            'Flow_Quality': model_info["conversation_flow_quality"],
            'Response_Time': model_info["avg_response_time"],
            'Cost': model_info["estimated_cost_usd"],
            'NDCG@5': model_info["ndcg_at_5"],
            'Recall@5': model_info["recall_at_5"]
        })
    
    df = pd.DataFrame(models_data)
    
    # Create comprehensive dashboard
    fig = plt.figure(figsize=(20, 16))
    
    # Define colors for SLM vs LLM
    colors = {'SLM': '#2E86AB', 'LLM': '#A23B72'}
    
    # 1. User Satisfaction Comparison
    ax1 = plt.subplot(3, 4, 1)
    bars = ax1.bar(df['Model'], df['User_Satisfaction'], 
                   color=[colors[t] for t in df['Type']])
    ax1.set_title('User Satisfaction Scores', fontweight='bold')
    ax1.set_ylabel('Satisfaction Score')
    ax1.tick_params(axis='x', rotation=45)
    ax1.set_ylim(0, 1.0)
    
    # Add value labels on bars
    for bar, value in zip(bars, df['User_Satisfaction']):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Response Time Comparison
    ax2 = plt.subplot(3, 4, 2)
    bars = ax2.bar(df['Model'], df['Response_Time'], 
                   color=[colors[t] for t in df['Type']])
    ax2.set_title('Response Time (seconds)', fontweight='bold')
    ax2.set_ylabel('Time (s)')
    ax2.tick_params(axis='x', rotation=45)
    
    for bar, value in zip(bars, df['Response_Time']):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{value:.2f}s', ha='center', va='bottom', fontsize=9)
    
    # 3. Cost Comparison
    ax3 = plt.subplot(3, 4, 3)
    bars = ax3.bar(df['Model'], df['Cost'], 
                   color=[colors[t] for t in df['Type']])
    ax3.set_title('Cost per Conversation (USD)', fontweight='bold')
    ax3.set_ylabel('Cost ($)')
    ax3.tick_params(axis='x', rotation=45)
    
    for bar, value in zip(bars, df['Cost']):
        ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.00005,
                f'${value:.4f}', ha='center', va='bottom', fontsize=9)
    
    # 4. NDCG@5 Comparison
    ax4 = plt.subplot(3, 4, 4)
    bars = ax4.bar(df['Model'], df['NDCG@5'], 
                   color=[colors[t] for t in df['Type']])
    ax4.set_title('NDCG@5 Performance', fontweight='bold')
    ax4.set_ylabel('NDCG@5 Score')
    ax4.tick_params(axis='x', rotation=45)
    ax4.set_ylim(0.95, 1.01)
    
    # 5. Quality Metrics Heatmap
    ax5 = plt.subplot(3, 4, (5, 6))
    quality_metrics = ['User_Satisfaction', 'Response_Coherence', 'Tool_Accuracy', 'Flow_Quality']
    heatmap_data = df.set_index('Model')[quality_metrics].T
    
    sns.heatmap(heatmap_data, annot=True, cmap='RdYlGn', center=0.7, 
                fmt='.3f', ax=ax5, cbar_kws={'label': 'Performance Score'})
    ax5.set_title('Quality Metrics Heatmap', fontweight='bold')
    ax5.set_xlabel('Models')
    
    # 6. SLM vs LLM Type Comparison
    ax6 = plt.subplot(3, 4, 7)
    
    # Calculate type averages
    slm_data = df[df['Type'] == 'SLM']
    llm_data = df[df['Type'] == 'LLM']
    
    type_comparison = {
        'Quality': [slm_data['User_Satisfaction'].mean(), llm_data['User_Satisfaction'].mean()],
        'Speed': [1/slm_data['Response_Time'].mean(), 1/llm_data['Response_Time'].mean()],  # Inverse for better visualization
        'Cost_Eff': [1/slm_data['Cost'].mean(), 1/llm_data['Cost'].mean()]  # Inverse for better visualization
    }
    
    x = np.arange(len(type_comparison))
    width = 0.35
    
    ax6.bar(x - width/2, [type_comparison['Quality'][0], type_comparison['Speed'][0]*0.1, type_comparison['Cost_Eff'][0]*0.001], 
           width, label='SLM Average', color=colors['SLM'])
    ax6.bar(x + width/2, [type_comparison['Quality'][1], type_comparison['Speed'][1]*0.1, type_comparison['Cost_Eff'][1]*0.001], 
           width, label='LLM Average', color=colors['LLM'])
    
    ax6.set_title('SLM vs LLM Type Comparison', fontweight='bold')
    ax6.set_ylabel('Normalized Score')
    ax6.set_xticks(x)
    ax6.set_xticklabels(['Quality', 'Speed*0.1', 'Cost_Eff*0.001'])
    ax6.legend()
    
    # 7. Cost vs Quality Trade-off
    ax7 = plt.subplot(3, 4, 8)
    scatter = ax7.scatter(df['Cost'], df['User_Satisfaction'], 
                         c=[colors[t] for t in df['Type']], s=100, alpha=0.7)
    
    # Add model labels
    for i, model in enumerate(df['Model']):
        ax7.annotate(model, (df['Cost'].iloc[i], df['User_Satisfaction'].iloc[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax7.set_title('Cost vs Quality Trade-off', fontweight='bold')
    ax7.set_xlabel('Cost per Conversation ($)')
    ax7.set_ylabel('User Satisfaction')
    
    # 8. Model Efficiency Matrix
    ax8 = plt.subplot(3, 4, 9)
    efficiency_score = (df['User_Satisfaction'] * 1000) / (df['Response_Time'] * df['Cost'] * 10000 + 1)  # Composite efficiency
    
    bars = ax8.bar(df['Model'], efficiency_score, 
                   color=[colors[t] for t in df['Type']])
    ax8.set_title('Overall Efficiency Score', fontweight='bold')
    ax8.set_ylabel('Efficiency Score')
    ax8.tick_params(axis='x', rotation=45)
    
    # 9. Performance Radar Chart (simplified as grouped bar)
    ax9 = plt.subplot(3, 4, 10)
    metrics_for_radar = ['User_Satisfaction', 'Response_Coherence', 'Tool_Accuracy', 'Flow_Quality']
    
    # Calculate averages by type
    slm_averages = [slm_data[metric].mean() for metric in metrics_for_radar]
    llm_averages = [llm_data[metric].mean() for metric in metrics_for_radar]
    
    x = np.arange(len(metrics_for_radar))
    width = 0.35
    
    ax9.bar(x - width/2, slm_averages, width, label='SLM Average', color=colors['SLM'])
    ax9.bar(x + width/2, llm_averages, width, label='LLM Average', color=colors['LLM'])
    
    ax9.set_title('Quality Metrics Comparison', fontweight='bold')
    ax9.set_ylabel('Score')
    ax9.set_xticks(x)
    ax9.set_xticklabels([m.replace('_', ' ') for m in metrics_for_radar], rotation=45)
    ax9.legend()
    ax9.set_ylim(0, 1.0)
    
    # 10. Time vs Quality Analysis
    ax10 = plt.subplot(3, 4, 11)
    scatter = ax10.scatter(df['Response_Time'], df['User_Satisfaction'], 
                          c=[colors[t] for t in df['Type']], s=100, alpha=0.7)
    
    for i, model in enumerate(df['Model']):
        ax10.annotate(model, (df['Response_Time'].iloc[i], df['User_Satisfaction'].iloc[i]),
                     xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax10.set_title('Response Time vs Quality', fontweight='bold')
    ax10.set_xlabel('Response Time (seconds)')
    ax10.set_ylabel('User Satisfaction')
    
    # 11. Summary Statistics
    ax11 = plt.subplot(3, 4, 12)
    ax11.axis('off')
    
    summary_text = f"""
    📊 SLM vs LLM Summary
    
    🏆 Best Quality: {df.loc[df['User_Satisfaction'].idxmax(), 'Model']}
    ⚡ Fastest: {df.loc[df['Response_Time'].idxmin(), 'Model']}
    💰 Most Cost-Effective: {df.loc[df['Cost'].idxmin(), 'Model']}
    
    Average Performance:
    SLM Quality: {slm_data['User_Satisfaction'].mean():.3f}
    LLM Quality: {llm_data['User_Satisfaction'].mean():.3f}
    
    SLM Speed: {slm_data['Response_Time'].mean():.2f}s
    LLM Speed: {llm_data['Response_Time'].mean():.2f}s
    
    SLM Cost: ${slm_data['Cost'].mean():.4f}
    LLM Cost: ${llm_data['Cost'].mean():.4f}
    
    💡 Key Insight:
    {'SLMs provide competitive quality' if slm_data['User_Satisfaction'].mean() > 0.7 else 'LLMs significantly outperform'}
    at much lower cost
    """
    
    ax11.text(0.1, 0.9, summary_text, transform=ax11.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.suptitle('SLM vs LLM MUSE Performance Analysis Dashboard', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95, hspace=0.4, wspace=0.3)
    
    # Save plot
    plt.savefig('/media/adityapachauri/second_drive/Muse/slm_llm_comparison_dashboard.png', 
                dpi=300, bbox_inches='tight')
    print("📊 SLM vs LLM comparison dashboard saved as 'slm_llm_comparison_dashboard.png'")
    plt.show()

def create_efficiency_analysis(analysis: dict):
    """Create detailed efficiency analysis visualization"""
    
    model_rankings = analysis["model_rankings"]["by_quality"]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Extract data
    models = []
    types = []
    quality = []
    speed = []
    cost = []
    
    for model_key, model_info in model_rankings:
        models.append(model_info["model_name"])
        types.append(model_info["model_type"])
        quality.append(model_info["user_satisfaction_score"])
        speed.append(model_info["avg_response_time"])
        cost.append(model_info["estimated_cost_usd"])
    
    colors = {'SLM': '#2E86AB', 'LLM': '#A23B72'}
    type_colors = [colors[t] for t in types]
    
    # 1. Quality vs Speed
    scatter1 = ax1.scatter(speed, quality, c=type_colors, s=150, alpha=0.7)
    for i, model in enumerate(models):
        ax1.annotate(model, (speed[i], quality[i]), xytext=(5, 5), 
                    textcoords='offset points', fontsize=9)
    ax1.set_xlabel('Response Time (seconds)')
    ax1.set_ylabel('User Satisfaction')
    ax1.set_title('Quality vs Speed Trade-off')
    ax1.grid(True, alpha=0.3)
    
    # 2. Quality vs Cost
    scatter2 = ax2.scatter(cost, quality, c=type_colors, s=150, alpha=0.7)
    for i, model in enumerate(models):
        ax2.annotate(model, (cost[i], quality[i]), xytext=(5, 5), 
                    textcoords='offset points', fontsize=9)
    ax2.set_xlabel('Cost per Conversation ($)')
    ax2.set_ylabel('User Satisfaction')
    ax2.set_title('Quality vs Cost Trade-off')
    ax2.grid(True, alpha=0.3)
    
    # 3. Speed vs Cost
    scatter3 = ax3.scatter(cost, speed, c=type_colors, s=150, alpha=0.7)
    for i, model in enumerate(models):
        ax3.annotate(model, (cost[i], speed[i]), xytext=(5, 5), 
                    textcoords='offset points', fontsize=9)
    ax3.set_xlabel('Cost per Conversation ($)')
    ax3.set_ylabel('Response Time (seconds)')
    ax3.set_title('Speed vs Cost Analysis')
    ax3.grid(True, alpha=0.3)
    
    # 4. Efficiency Index (Quality per dollar per second)
    efficiency = np.array(quality) / (np.array(speed) * np.array(cost) + 0.0001)
    bars = ax4.bar(models, efficiency, color=type_colors)
    ax4.set_ylabel('Efficiency Index')
    ax4.set_title('Overall Efficiency Ranking')
    ax4.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, value in zip(bars, efficiency):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(efficiency)*0.01,
                f'{value:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Create legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors['SLM'], label='SLM'),
                      Patch(facecolor=colors['LLM'], label='LLM')]
    fig.legend(handles=legend_elements, loc='upper right')
    
    plt.suptitle('SLM vs LLM Efficiency Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/media/adityapachauri/second_drive/Muse/slm_llm_efficiency_analysis.png', 
                dpi=300, bbox_inches='tight')
    print("📈 Efficiency analysis saved as 'slm_llm_efficiency_analysis.png'")
    plt.show()

def main():
    """Main function to create SLM vs LLM visualizations"""
    print("🎨 Creating SLM vs LLM Comparison Visualizations...")
    
    # Load analysis results
    analysis_file = '/media/adityapachauri/second_drive/Muse/slm_llm_analysis.json'
    analysis = load_slm_llm_results(analysis_file)
    
    print("📊 Creating comprehensive comparison dashboard...")
    create_slm_llm_comparison_dashboard(analysis)
    
    print("📈 Creating efficiency analysis...")
    create_efficiency_analysis(analysis)
    
    print("✅ All SLM vs LLM visualizations created successfully!")
    print("📁 Generated files:")
    print("   - slm_llm_comparison_dashboard.png")
    print("   - slm_llm_efficiency_analysis.png")

if __name__ == "__main__":
    main()
