import streamlit as st
import numpy as np
import pandas as pd
import torch
import cv2
import os
import sys
import json
import plotly.graph_objects as go
import plotly.express as px
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "Autonomous Car — Control Panel",
    page_icon   = "🚗",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Load model (cached) ───────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    from models.fusion_model import FusionModel
    model = FusionModel()
    ckpt  = torch.load(
        "checkpoints_v2/best_model.pth",
        map_location="cpu",
        weights_only=True
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt

@st.cache_resource
def load_onnx():
    import onnxruntime as ort
    sess = ort.InferenceSession(
        "checkpoints_v2/fusion_model.onnx",
        providers=["CPUExecutionProvider"]
    )
    return sess

# ── Load training history (cached) ────────────────────────────────────────────
@st.cache_data
def load_history():
    with open("checkpoints_v2/history.json") as f:
        return json.load(f)

@st.cache_data
def load_eval():
    with open("checkpoints_v2/eval_metrics.json") as f:
        return json.load(f)

@st.cache_data
def load_episodes():
    eps = sorted(glob("data_clean/processed/episode_*"))
    return eps

# ── Inference helpers ─────────────────────────────────────────────────────────
def predict_pytorch(model, frame_np, sensor_np):
    frame  = torch.from_numpy(
        cv2.resize(frame_np, (128,128)).astype(np.float32) / 255.0
    ).permute(2,0,1).unsqueeze(0)
    sensor = torch.from_numpy(sensor_np.astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        out = model(frame, sensor).squeeze().numpy()
    return out

def predict_onnx(sess, frame_np, sensor_np):
    frame  = cv2.resize(frame_np, (128,128)).astype(np.float32) / 255.0
    frame  = np.transpose(frame, (2,0,1))[None]
    sensor = sensor_np.astype(np.float32)[None]
    out    = sess.run(["action"], {"frame": frame, "sensor": sensor})[0]
    return out.squeeze()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title("AV Control Panel")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Training", "Evaluation", "Episode Replay", "Live Inference"]
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Model:** FusionModel v2")
st.sidebar.markdown("**Data:** 90,155 IDM steps")
st.sidebar.markdown("**Val Loss:** 0.00043")
st.sidebar.markdown("**ONNX Speed:** 226 Hz")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if page == "Overview":
    st.title("Autonomous Car Simulation — Phase 1")
    st.markdown("**Behavioral Cloning with Sensor Fusion | EfficientNet-B0 + SensorMLP**")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Val Loss",       "0.00043",  "-95.6% vs keyboard data")
    c2.metric("Sensor Fusion",  "+81.5%",   "vs camera-only baseline")
    c3.metric("ONNX Latency",   "4.42 ms",  "226 Hz on CPU")
    c4.metric("Model Size",     "18.7 MB",  "-67% vs PyTorch .pth")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Architecture")
        st.markdown("""
| Branch | Details |
|---|---|
| **Visual** | EfficientNet-B0 → FC(1280→512) |
| **Sensor** | FC(19→64→128→128) |
| **Fusion** | Cat(640) → FC→FC → 2 outputs |
| **Output** | steering ∈ (-1,1), accel ∈ (-1,1) |
        """)

    with col2:
        st.subheader("Training Setup")
        st.markdown("""
| Setting | Value |
|---|---|
| Optimizer | Adam (head=1e-4, backbone=1e-5) |
| Scheduler | CosineAnnealingLR |
| Unfreeze epoch | 5 |
| Batch size | 32 |
| Early stopping | patience=15 |
        """)

    st.markdown("---")
    st.subheader("Failure Analysis Summary")
    st.error("**Primary:** Moderate turn underestimation (40-60% undershoot on |steer|=0.1-0.4)")
    st.warning("**Secondary:** Speed distribution mismatch (IDM trained at 7-8 m/s, agent ramps slowly)")
    st.info("**Fix:** Turn-weighted loss + top-down camera + Phase 2 RL")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: TRAINING
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Training":
    st.title("Training History")
    history = load_history()
    train_l = history["train_loss"]
    val_l   = history["val_loss"]
    epochs  = list(range(1, len(train_l)+1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=train_l, name="Train Loss",
                             line=dict(color="#4fc3f7", width=2)))
    fig.add_trace(go.Scatter(x=epochs, y=val_l,   name="Val Loss",
                             line=dict(color="#ef5350", width=2)))
    fig.add_vline(x=6,  line_dash="dash", line_color="gray",
                  annotation_text="Backbone unfrozen")
    best_ep = val_l.index(min(val_l)) + 1
    fig.add_vline(x=best_ep, line_dash="dash", line_color="green",
                  annotation_text=f"Best val={min(val_l):.5f}")
    fig.update_layout(
        yaxis_type="log", yaxis_title="MSE Loss (log)",
        xaxis_title="Epoch", height=450,
        legend=dict(x=0.7, y=0.95),
        plot_bgcolor="#1a1d27", paper_bgcolor="#0f1117",
        font_color="white"
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Best Val Loss", f"{min(val_l):.5f}", f"Epoch {best_ep}")
    col2.metric("Final Train Loss", f"{train_l[-1]:.5f}")
    col3.metric("Total Epochs", len(train_l))

    st.markdown("---")
    st.subheader("Ablation: Sensor Fusion vs Camera-Only")
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=["Camera-Only", "Fusion (Camera+Sensor)"],
        y=[0.05322, 0.00043],
        marker_color=["#ef5350", "#66bb6a"],
        text=["0.05322", "0.00043"],
        textposition="outside"
    ))
    fig2.update_layout(
        yaxis_title="Val Loss (MSE)", height=350,
        plot_bgcolor="#1a1d27", paper_bgcolor="#0f1117",
        font_color="white",
        title="81.5% improvement from sensor fusion"
    )
    st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Evaluation":
    st.title("Evaluation Metrics")
    eval_data = load_eval()
    results   = eval_data["episode_results"]

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Success Rate",    "0/20 (0%)")
    c2.metric("Avg Distance",    f"{eval_data['avg_distance_m']:.0f}m")
    c3.metric("Avg Velocity",    f"{eval_data['avg_velocity_ms']:.2f} m/s")
    c4.metric("Incident/km",     f"{eval_data['incident_rate_per_km']:.2f}")

    st.markdown("---")

    # Distance bar chart
    df = pd.DataFrame(results)
    colors = ["#ef5350" if d < 50 else "#ffa726" if d < 150 else "#66bb6a"
              for d in df["distance_m"]]

    fig = go.Figure(go.Bar(
        x=df["episode"], y=df["distance_m"],
        marker_color=colors, name="Distance"
    ))
    fig.add_hline(y=df["distance_m"].mean(), line_dash="dash",
                  line_color="#4fc3f7",
                  annotation_text=f"Mean={df['distance_m'].mean():.0f}m")
    fig.update_layout(
        xaxis_title="Episode", yaxis_title="Distance (m)",
        height=380, title="Distance Traveled per Episode",
        plot_bgcolor="#1a1d27", paper_bgcolor="#0f1117", font_color="white"
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        fig2 = px.scatter(
            df, x="steer_std", y="distance_m", color="reward",
            color_continuous_scale="plasma", size_max=15,
            labels={"steer_std": "Steering Std", "distance_m": "Distance (m)"},
            title="Turn Activity vs Distance"
        )
        fig2.update_layout(
            height=350,
            plot_bgcolor="#1a1d27", paper_bgcolor="#0f1117", font_color="white"
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        fig3 = px.histogram(
            df, x="avg_velocity", nbins=10,
            title="Avg Velocity Distribution",
            labels={"avg_velocity": "Avg Velocity (m/s)"},
            color_discrete_sequence=["#4fc3f7"]
        )
        fig3.add_vline(x=5.0, line_dash="dash", line_color="#66bb6a",
                       annotation_text="Target 5.0")
        fig3.update_layout(
            height=350,
            plot_bgcolor="#1a1d27", paper_bgcolor="#0f1117", font_color="white"
        )
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")
    st.subheader("Episode Details")
    st.dataframe(
        df[["episode","seed","status","steps","reward",
            "avg_velocity","distance_m","steer_std"]].round(3),
        use_container_width=True
    )

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: EPISODE REPLAY
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Episode Replay":
    st.title("Episode Replay")
    st.markdown("Browse frames and sensor data from collected episodes.")

    eps      = load_episodes()
    ep_names = [os.path.basename(e) for e in eps]
    selected = st.selectbox("Select episode", ep_names)
    ep_dir   = os.path.join("data_clean/processed", selected)

    act = pd.read_csv(os.path.join(ep_dir, "actions.csv"))
    sen = pd.read_csv(os.path.join(ep_dir, "sensors.csv"))

    st.markdown(f"**Steps:** {len(act)} | "
                f"**Steer range:** [{act['steering'].min():.3f}, {act['steering'].max():.3f}] | "
                f"**Accel mean:** {act['acceleration'].mean():.3f}")

    # Frame slider
    frame_idx = st.slider("Frame", 0, len(act)-1, 0)
    timestep  = int(act["timestep"].iloc[frame_idx])
    frame_path = os.path.join(ep_dir, "frames", f"{timestep:06d}.npy")

    col1, col2 = st.columns([1, 2])

    with col1:
        if os.path.exists(frame_path):
            frame = np.load(frame_path)
            st.image(frame, caption=f"Frame {timestep}", use_container_width=True)
        st.markdown("**Actions at this step:**")
        steer = act["steering"].iloc[frame_idx]
        accel = act["acceleration"].iloc[frame_idx]
        st.metric("Steering",     f"{steer:.4f}")
        st.metric("Acceleration", f"{accel:.4f}")
        st.markdown("**Sensor state:**")
        sensor_vals = sen.iloc[frame_idx][[f"s{j}" for j in range(19)]].values
        st.bar_chart(pd.DataFrame({"value": sensor_vals},
                                   index=[f"s{j}" for j in range(19)]))

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=act.index, y=act["steering"],
            name="Steering", line=dict(color="#4fc3f7")
        ))
        fig.add_trace(go.Scatter(
            x=act.index, y=act["acceleration"],
            name="Acceleration", line=dict(color="#ffa726")
        ))
        fig.add_vline(x=frame_idx, line_dash="dash", line_color="white",
                      annotation_text="current")
        fig.update_layout(
            height=300, title="Action Trace",
            plot_bgcolor="#1a1d27", paper_bgcolor="#0f1117",
            font_color="white", xaxis_title="Timestep"
        )
        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=sen.index, y=sen["velocity"],
            name="Velocity", line=dict(color="#66bb6a")
        ))
        fig2.add_vline(x=frame_idx, line_dash="dash", line_color="white")
        fig2.update_layout(
            height=250, title="Velocity Trace",
            plot_bgcolor="#1a1d27", paper_bgcolor="#0f1117",
            font_color="white", xaxis_title="Timestep"
        )
        st.plotly_chart(fig2, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: LIVE INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "Live Inference":
    st.title("Live Inference Demo")
    st.markdown("Run model inference on frames from the dataset.")

    backend = st.radio("Backend", ["PyTorch (GPU/CPU)", "ONNX (CPU)"],
                       horizontal=True)

    eps      = load_episodes()
    ep_names = [os.path.basename(e) for e in eps]
    selected = st.selectbox("Select episode", ep_names)
    ep_dir   = os.path.join("data_clean/processed", selected)

    act        = pd.read_csv(os.path.join(ep_dir, "actions.csv"))
    sen        = pd.read_csv(os.path.join(ep_dir, "sensors.csv"))
    frame_idx  = st.slider("Frame", 0, len(act)-1, 0)
    timestep   = int(act["timestep"].iloc[frame_idx])
    frame_path = os.path.join(ep_dir, "frames", f"{timestep:06d}.npy")

    if os.path.exists(frame_path):
        frame_np  = np.load(frame_path)
        sensor_np = sen.iloc[frame_idx][[f"s{j}" for j in range(19)]].values

        if backend == "PyTorch (GPU/CPU)":
            model, ckpt = load_model()
            pred = predict_pytorch(model, frame_np, sensor_np)
        else:
            sess = load_onnx()
            pred = predict_onnx(sess, frame_np, sensor_np)

        gt_steer = float(act["steering"].iloc[frame_idx])
        gt_accel = float(act["acceleration"].iloc[frame_idx])

        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(frame_np, caption="Input frame", use_container_width=True)

        with col2:
            st.markdown(f"**Backend:** `{backend}`")
            st.markdown("---")

            c1, c2 = st.columns(2)
            c1.metric("Predicted Steering",     f"{pred[0]:.4f}",
                      f"{pred[0]-gt_steer:+.4f} vs GT")
            c2.metric("Predicted Acceleration",  f"{pred[1]:.4f}",
                      f"{pred[1]-gt_accel:+.4f} vs GT")
            c1.metric("Ground Truth Steering",   f"{gt_steer:.4f}")
            c2.metric("Ground Truth Accel",       f"{gt_accel:.4f}")

            st.markdown("---")
            # Gauge charts
            fig = go.Figure()
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=float(pred[0]),
                title={"text": "Predicted Steering"},
                gauge={"axis": {"range": [-1, 1]},
                       "bar":  {"color": "#4fc3f7"},
                       "steps": [
                           {"range": [-1, -0.3], "color": "#ef5350"},
                           {"range": [-0.3, 0.3],"color": "#1a1d27"},
                           {"range": [0.3, 1],   "color": "#66bb6a"},
                       ]},
                domain={"x": [0, 0.5], "y": [0, 1]}
            ))
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=float(pred[1]),
                title={"text": "Predicted Accel"},
                gauge={"axis": {"range": [-1, 1]},
                       "bar":  {"color": "#ffa726"}},
                domain={"x": [0.5, 1], "y": [0, 1]}
            ))
            fig.update_layout(
                height=300, paper_bgcolor="#0f1117", font_color="white"
            )
            st.plotly_chart(fig, use_container_width=True)