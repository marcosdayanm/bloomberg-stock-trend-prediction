"""
Minimal Stock Prediction Dashboard
Live testing visualization for MSFT stock predictions
"""

import torch
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
import time

from src.model.config import ModelConfig
from src.model.model import CNNBiLSTMModel
from pathlib import PosixPath, WindowsPath
from torch.utils.data import DataLoader, TensorDataset


# Page config
st.set_page_config(
    page_title="MSFT Stock Predictions",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)


@st.cache_resource
def load_model(model_path: str):
    """Load the trained model"""
    config = ModelConfig()
    torch.serialization.add_safe_globals([PosixPath, WindowsPath])
    
    model = CNNBiLSTMModel.load_from_checkpoint(
        str(model_path),
        config=config,
        strict=False
    )
    model.eval()
    return model, config


@st.cache_data
def load_test_data(x_path: str, y_path: str):
    """Load test dataset"""
    X = np.load(x_path)
    y = np.load(y_path)
    
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    return dataloader, X.shape[0]


def create_confusion_matrix(predictions_history):
    """Create confusion matrix from predictions"""
    if len(predictions_history) == 0:
        return None
    
    df = pd.DataFrame(predictions_history)
    
    # Calculate confusion matrix values
    tn = len(df[(df['actual'] == 0) & (df['predicted'] == 0)])
    fp = len(df[(df['actual'] == 0) & (df['predicted'] == 1)])
    fn = len(df[(df['actual'] == 1) & (df['predicted'] == 0)])
    tp = len(df[(df['actual'] == 1) & (df['predicted'] == 1)])
    
    confusion_matrix = [[tn, fp], [fn, tp]]
    
    fig = go.Figure(data=go.Heatmap(
        z=confusion_matrix,
        x=['Pred DOWN', 'Pred UP'],
        y=['Actual DOWN', 'Actual UP'],
        text=confusion_matrix,
        texttemplate="%{text}",
        textfont={"size": 24, "color": "white"},
        colorscale='RdYlGn',
        showscale=False,
        hovertemplate='Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': "Confusion Matrix",
            'x': 0.5,
            'xanchor': 'center'
        },
        height=350,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis={'side': 'bottom'}
    )
    
    return fig


def create_metrics_chart(predictions_history):
    """Create accuracy by class chart"""
    if len(predictions_history) == 0:
        return None
    
    df = pd.DataFrame(predictions_history)
    
    # Count by class
    down_correct = len(df[(df['actual'] == 0) & (df['correct'] == True)])
    down_total = len(df[df['actual'] == 0])
    up_correct = len(df[(df['actual'] == 1) & (df['correct'] == True)])
    up_total = len(df[df['actual'] == 1])
    
    total_correct = len(df[df['correct'] == True])
    total = len(df)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=['DOWN', 'UP', 'TOTAL'],
        y=[down_correct/down_total*100 if down_total > 0 else 0,
           up_correct/up_total*100 if up_total > 0 else 0,
           total_correct/total*100],
        marker_color=['#EF476F', '#06D6A0', '#118AB2'],
        text=[f"{down_correct}/{down_total}", 
              f"{up_correct}/{up_total}", 
              f"{total_correct}/{total}"],
        textposition='auto',
        textfont=dict(size=14, color='white'),
        hovertemplate='%{x}: %{y:.1f}%<br>%{text}<extra></extra>'
    ))
    
    fig.add_hline(y=50, line_dash="dash", line_color="gray", 
                  annotation_text="50% Baseline", annotation_position="right")
    
    fig.update_layout(
        title={
            'text': "Accuracy by Class",
            'x': 0.5,
            'xanchor': 'center'
        },
        yaxis_title="Accuracy (%)",
        height=350,
        showlegend=False,
        yaxis=dict(range=[0, 100]),
        margin=dict(l=20, r=20, t=60, b=20)
    )
    
    return fig


def main():
    st.title("📈 MSFT Stock Prediction - Live Testing")
    
    # Simple config at top
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("**Model:** CNN-BiLSTM | **Task:** Binary Classification")
    
    with col2:
        animation_speed = st.slider(
            "Speed (sec)",
            min_value=0.0,
            max_value=1.0,
            value=0.3,
            step=0.05
        )
    
    with col3:
        auto_play = st.checkbox("Auto Play", value=True)
    
    # Fixed paths
    model_path = "pretrained_models/best_model_v2.1.ckpt"
    x_path = "datasets/npy/test_X.npy"
    y_path = "datasets/npy/test_y.npy"
    
    # Load resources
    try:
        with st.spinner("🔄 Loading model..."):
            model, config = load_model(model_path)
            test_dataloader, n_samples = load_test_data(x_path, y_path)
        
        st.success(f"✅ Ready! {n_samples} test samples loaded")
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return
    
    # Initialize session state
    if 'current_idx' not in st.session_state:
        st.session_state.current_idx = 0
        st.session_state.predictions_history = []
        st.session_state.total_correct = 0
        st.session_state.running = False
    
    # Control buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("▶️ RUN" if not st.session_state.running else "⏸️ PAUSE", 
                     use_container_width=True, type="primary"):
            st.session_state.running = not st.session_state.running
    
    with col2:
        if st.button("🔄 RESET", use_container_width=True):
            st.session_state.current_idx = 0
            st.session_state.predictions_history = []
            st.session_state.total_correct = 0
            st.session_state.running = False
            st.rerun()
    
    with col3:
        if st.button("⏩ FINISH", use_container_width=True):
            st.session_state.current_idx = n_samples
            st.session_state.running = False
    
    # Progress bar
    progress = st.session_state.current_idx / n_samples
    st.progress(progress, text=f"Progress: {st.session_state.current_idx}/{n_samples}")
    
    st.markdown("---")
    
    # Main layout: Stats on left, Current prediction on right
    left_col, right_col = st.columns([1, 1])
    
    with left_col:
        st.subheader("📊 Overall Results")
        
        if len(st.session_state.predictions_history) > 0:
            accuracy = st.session_state.total_correct / len(st.session_state.predictions_history)
            
            # Big metrics
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric(
                    "Accuracy",
                    f"{accuracy:.1%}",
                    delta=f"{st.session_state.total_correct}/{len(st.session_state.predictions_history)}"
                )
            with metric_col2:
                wrong = len(st.session_state.predictions_history) - st.session_state.total_correct
                st.metric(
                    "Errors",
                    wrong,
                    delta=f"{(1-accuracy):.1%}",
                    delta_color="inverse"
                )
            
            # Confusion Matrix
            cm_fig = create_confusion_matrix(st.session_state.predictions_history)
            if cm_fig:
                st.plotly_chart(cm_fig, width='stretch')
            
            # Metrics by class
            metrics_fig = create_metrics_chart(st.session_state.predictions_history)
            if metrics_fig:
                st.plotly_chart(metrics_fig, width='stretch')
        
        else:
            st.info("👈 Click RUN to start predictions")
    
    with right_col:
        st.subheader("🎯 Current Prediction")
        
        if st.session_state.current_idx < n_samples:
            device = next(model.parameters()).device
            
            # Get current sample
            sample_idx = st.session_state.current_idx
            
            # Get batch
            for idx, (x, y) in enumerate(test_dataloader):
                if idx == sample_idx:
                    current_x = x
                    current_y = y
                    break
            
            # Make prediction
            with torch.no_grad():
                current_x = current_x.to(device)
                current_y = current_y.to(device)
                
                outputs = model(current_x)
                probs = torch.softmax(outputs, dim=1)
                pred = torch.argmax(outputs, dim=1)
                
                y_true = torch.argmax(current_y, dim=1) if current_y.dim() > 1 else current_y
                
                pred_class = int(pred[0].item())
                true_class = int(y_true[0].item())
                confidence = float(probs[0, pred_class].item())
                
                is_correct = (pred_class == true_class)
                
                class_names = ['DOWN ⬇️', 'UP ⬆️']
                colors = ['#EF476F', '#06D6A0']
            
            # Display
            st.markdown(f"### Sample #{sample_idx + 1}/{n_samples}")
            
            # Result
            if is_correct:
                st.success(f"✅ CORRECT - Confidence: {confidence:.1%}")
            else:
                st.error(f"❌ WRONG - Confidence: {confidence:.1%}")
            
            # Prediction vs Actual
            pred_col, actual_col = st.columns(2)
            
            with pred_col:
                st.markdown(f"""
                    <div style="background: {colors[pred_class]}; padding: 20px; 
                                border-radius: 10px; text-align: center;">
                        <h3 style="color: white; margin: 0;">PREDICTED</h3>
                        <h1 style="color: white; margin: 10px 0; font-size: 48px;">
                            {class_names[pred_class]}
                        </h1>
                        <p style="color: white; font-size: 18px; margin: 0;">
                            {confidence:.1%} confidence
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            
            with actual_col:
                st.markdown(f"""
                    <div style="background: {colors[true_class]}; padding: 20px; 
                                border-radius: 10px; text-align: center;">
                        <h3 style="color: white; margin: 0;">ACTUAL</h3>
                        <h1 style="color: white; margin: 10px 0; font-size: 48px;">
                            {class_names[true_class]}
                        </h1>
                    </div>
                """, unsafe_allow_html=True)
            
            # Probability bars
            st.markdown("#### Probability Distribution")
            prob_fig = go.Figure(data=[
                go.Bar(
                    x=class_names,
                    y=[float(probs[0, 0].item()), float(probs[0, 1].item())],
                    marker_color=colors,
                    text=[f"{probs[0, 0].item():.1%}", f"{probs[0, 1].item():.1%}"],
                    textposition='auto',
                    textfont=dict(size=16, color='white'),
                )
            ])
            prob_fig.update_layout(
                height=200,
                yaxis=dict(range=[0, 1], title="Probability"),
                showlegend=False,
                margin=dict(l=20, r=20, t=20, b=20)
            )
            st.plotly_chart(prob_fig, width='stretch')
            
            # Auto advance
            if st.session_state.running or st.button("✅ Next →", use_container_width=True, type="primary"):
                # Add to history
                st.session_state.predictions_history.append({
                    'sample': sample_idx + 1,
                    'predicted': pred_class,
                    'actual': true_class,
                    'correct': is_correct,
                    'confidence': confidence
                })
                
                if is_correct:
                    st.session_state.total_correct += 1
                
                st.session_state.current_idx += 1
                
                if auto_play and st.session_state.running and animation_speed > 0:
                    time.sleep(animation_speed)
                
                st.rerun()
        
        else:
            # Completed
            if len(st.session_state.predictions_history) > 0:
                final_accuracy = st.session_state.total_correct / len(st.session_state.predictions_history)
                
                st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                padding: 40px; border-radius: 15px; text-align: center;">
                        <h1 style="color: white; margin: 0;">🎉 COMPLETED!</h1>
                        <h2 style="color: white; margin: 20px 0;">
                            Final Accuracy: {final_accuracy:.2%}
                        </h2>
                        <h3 style="color: white; margin: 0;">
                            {st.session_state.total_correct} / {len(st.session_state.predictions_history)} correct
                        </h3>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.info("No predictions yet. Click RESET to start.")


if __name__ == "__main__":
    main()
