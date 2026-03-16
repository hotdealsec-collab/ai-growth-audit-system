import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(
    page_title="AI-assisted Growth Audit System v0.4",
    layout="wide"
)

# --------------------------------------------------
# Custom style
# --------------------------------------------------
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
.small-note {
    color: #6b7280;
    font-size: 0.9rem;
}
.info-card {
    background-color: #111827;
    padding: 1rem 1.2rem;
    border-radius: 16px;
    border: 1px solid #1f2937;
    margin-bottom: 1rem;
}
.action-card {
    background-color: #0f172a;
    padding: 1rem 1.2rem;
    border-radius: 16px;
    border: 1px solid #1f2937;
    min-height: 220px;
}
.card-title {
    font-size: 1rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}
.card-subtle {
    color: #9ca3af;
    font-size: 0.85rem;
    margin-bottom: 0.75rem;
}
.mono-box {
    background-color: #111827;
    color: #f9fafb;
    padding: 1rem 1.2rem;
    border-radius: 16px;
    border: 1px solid #1f2937;
    font-family: monospace;
    font-size: 0.95rem;
    line-height: 1.8;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Required columns
# --------------------------------------------------
REQUIRED_COLUMNS = [
    "channel",
    "campaign",
    "os",
    "spend",
    "installs",
    "activated_users",
    "d1_retention",
    "d3_retention",
    "d7_retention",
    "revenue",
    "skan_only",
    "strategic_channel",
    "period_start",
    "period_end",
]

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def safe_divide(a, b):
    if pd.isna(a) or pd.isna(b) or b == 0:
        return np.nan
    return a / b

def normalize_bool(value):
    if pd.isna(value):
        return False
    s = str(value).strip().lower()
    return s in ["true", "1", "yes", "y"]

def rate_relative_low_is_good(value, avg_value):
    if pd.isna(value) or pd.isna(avg_value):
        return "Unknown"
    if value <= avg_value * 0.85:
        return "Strong"
    elif value <= avg_value * 1.15:
        return "Moderate"
    return "Weak"

def rate_relative_high_is_good(value, avg_value):
    if pd.isna(value) or pd.isna(avg_value):
        return "Unknown"
    if value >= avg_value * 1.15:
        return "Strong"
    elif value >= avg_value * 0.85:
        return "Moderate"
    return "Weak"

def os_tracking_risk(os_value):
    if str(os_value).strip().lower() == "ios":
        return "ATT Risk"
    if str(os_value).strip().lower() == "android":
        return "Low Risk"
    return "Unknown"

def map_score(value):
    score_map = {
        "Strong": 100,
        "Moderate": 60,
        "Weak": 30,
        "Risky": 20,
        "Unknown": 50,
    }
    return score_map.get(value, 50)

def score_category(score):
    if pd.isna(score):
        return "Unknown"
    if score >= 80:
        return "Healthy Growth"
    if score >= 60:
        return "Optimization Opportunity"
    if score >= 40:
        return "Structural Growth Issue"
    return "Critical Growth Risk"

# --------------------------------------------------
# Core engine
# --------------------------------------------------
def run_growth_audit_v4(df: pd.DataFrame):
    required_columns = REQUIRED_COLUMNS

    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    numeric_cols = [
        "spend", "installs", "activated_users",
        "d1_retention", "d3_retention", "d7_retention", "revenue",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["d1_retention", "d3_retention", "d7_retention"]:
        df[col] = df[col].clip(lower=0, upper=0.9999)

    df["skan_only"] = df["skan_only"].apply(normalize_bool)
    df["strategic_channel"] = df["strategic_channel"].apply(normalize_bool)
    df["os"] = df["os"].astype(str).str.strip()

    # Core metrics
    df["cpi"] = df.apply(lambda x: safe_divide(x["spend"], x["installs"]), axis=1)
    df["activation_rate"] = df.apply(lambda x: safe_divide(x["activated_users"], x["installs"]), axis=1)
    df["arpu"] = df.apply(lambda x: safe_divide(x["revenue"], x["installs"]), axis=1)

    df["d1_to_d3_drop"] = df["d1_retention"] - df["d3_retention"]
    df["d3_to_d7_drop"] = df["d3_retention"] - df["d7_retention"]
    df["d1_to_d7_drop"] = df["d1_retention"] - df["d7_retention"]

    df["early_signal_score"] = (
        df["d1_retention"] * 0.2 +
        df["d3_retention"] * 0.3 +
        df["d7_retention"] * 0.5
    ) * 100

    def retention_curve_quality(row):
        if pd.isna(row["d1_retention"]) or pd.isna(row["d7_retention"]) or row["d1_retention"] <= 0:
            return np.nan
        return row["d7_retention"] / row["d1_retention"]

    df["retention_curve_quality"] = df.apply(retention_curve_quality, axis=1)

    def estimate_ltv_v2(row):
        if pd.isna(row["arpu"]):
            return np.nan
        weighted_retention_factor = (
            row["d1_retention"] * 0.2 +
            row["d3_retention"] * 0.3 +
            row["d7_retention"] * 0.5
        )
        if pd.isna(weighted_retention_factor) or weighted_retention_factor <= 0:
            return np.nan
        return row["arpu"] * (1 + weighted_retention_factor * 5)

    df["estimated_ltv"] = df.apply(estimate_ltv_v2, axis=1)

    def predict_ltv_v2(row):
        if pd.isna(row["estimated_ltv"]):
            return np.nan
        stability_bonus = 1.0
        if pd.notna(row["retention_curve_quality"]):
            stability_bonus += row["retention_curve_quality"] * 0.2
        return row["estimated_ltv"] * stability_bonus

    df["predicted_ltv"] = df.apply(predict_ltv_v2, axis=1)

    def estimate_payback(cpi, arpu):
        if pd.isna(cpi) or pd.isna(arpu) or arpu <= 0:
            return np.nan
        return cpi / arpu

    df["payback_period"] = df.apply(lambda x: estimate_payback(x["cpi"], x["arpu"]), axis=1)

    avg_cpi = df["cpi"].mean(skipna=True)
    avg_arpu = df["arpu"].mean(skipna=True)

    def score_traffic_efficiency(cpi):
        return rate_relative_low_is_good(cpi, avg_cpi)

    def score_activation_efficiency(activation_rate):
        if pd.isna(activation_rate):
            return "Unknown"
        if activation_rate >= 0.50:
            return "Strong"
        elif activation_rate >= 0.30:
            return "Moderate"
        return "Weak"

    def score_retention_stability(d7_retention):
        if pd.isna(d7_retention):
            return "Unknown"
        if d7_retention >= 0.25:
            return "Strong"
        elif d7_retention >= 0.15:
            return "Moderate"
        return "Weak"

    def score_early_signal(early_signal_score):
        if pd.isna(early_signal_score):
            return "Unknown"
        if early_signal_score >= 30:
            return "Strong"
        elif early_signal_score >= 18:
            return "Moderate"
        return "Weak"

    def score_curve_quality(curve_quality):
        if pd.isna(curve_quality):
            return "Unknown"
        if curve_quality >= 0.45:
            return "Strong"
        elif curve_quality >= 0.30:
            return "Moderate"
        return "Weak"

    def score_revenue_efficiency(arpu):
        return rate_relative_high_is_good(arpu, avg_arpu)

    def score_payback_health(payback_period):
        if pd.isna(payback_period):
            return "Unknown"
        if payback_period <= 1.5:
            return "Strong"
        elif payback_period <= 3.0:
            return "Moderate"
        return "Risky"

    df["traffic_efficiency"] = df["cpi"].apply(score_traffic_efficiency)
    df["activation_efficiency"] = df["activation_rate"].apply(score_activation_efficiency)
    df["retention_stability"] = df["d7_retention"].apply(score_retention_stability)
    df["early_signal_health"] = df["early_signal_score"].apply(score_early_signal)
    df["curve_quality_health"] = df["retention_curve_quality"].apply(score_curve_quality)
    df["revenue_efficiency"] = df["arpu"].apply(score_revenue_efficiency)
    df["payback_health"] = df["payback_period"].apply(score_payback_health)

    def detect_primary_bottleneck_v2(row):
        if row["d1_retention"] < 0.20:
            return "Weak Day-1 fit"
        if row["d1_to_d3_drop"] > 0.20:
            return "Early activation decay"
        if row["d3_to_d7_drop"] > 0.15:
            return "Structural retention decay"
        if row["activation_efficiency"] == "Weak":
            return "Activation friction"
        if row["revenue_efficiency"] == "Weak" and row["retention_stability"] in ["Moderate", "Strong"]:
            return "Monetization weakness"
        if row["payback_health"] == "Risky":
            return "Capital inefficiency"
        if row["traffic_efficiency"] == "Weak" and row["retention_stability"] == "Weak":
            return "Acquisition inefficiency"
        return "No critical bottleneck detected"

    df["primary_bottleneck"] = df.apply(detect_primary_bottleneck_v2, axis=1)

    def growth_risk_level_v2(row):
        weak_signals = 0
        diagnostic_fields = [
            row["traffic_efficiency"], row["activation_efficiency"],
            row["retention_stability"], row["early_signal_health"],
            row["curve_quality_health"], row["revenue_efficiency"],
        ]
        weak_signals += sum(1 for x in diagnostic_fields if x == "Weak")
        weak_signals += 1 if row["payback_health"] == "Risky" else 0

        if row["primary_bottleneck"] in [
            "Weak Day-1 fit", "Early activation decay", "Structural retention decay"
        ] and row["payback_health"] == "Risky":
            return "High"
        if weak_signals >= 3:
            return "High"
        if weak_signals >= 1:
            return "Medium"
        return "Low"

    df["growth_risk_level"] = df.apply(growth_risk_level_v2, axis=1)

    def allocation_suggestion_v2(row):
        if (
            row["early_signal_health"] == "Strong"
            and row["payback_health"] == "Strong"
            and row["revenue_efficiency"] in ["Moderate", "Strong"]
        ):
            return "Scale"
        if row["primary_bottleneck"] in [
            "Weak Day-1 fit", "Early activation decay", "Activation friction"
        ]:
            return "Optimize"
        if row["primary_bottleneck"] in [
            "Structural retention decay", "Monetization weakness"
        ] and row["payback_health"] != "Risky":
            return "Maintain"
        if row["payback_health"] == "Risky" and row["growth_risk_level"] in ["Medium", "High"]:
            return "Reduce"
        return "Maintain"

    df["allocation_suggestion"] = df.apply(allocation_suggestion_v2, axis=1)

    # Growth health
    df["traffic_score"] = df["traffic_efficiency"].map(map_score)
    df["activation_score"] = df["activation_efficiency"].map(map_score)
    df["early_signal_score_norm"] = df["early_signal_health"].map(map_score)
    df["retention_score"] = df["retention_stability"].map(map_score)
    df["revenue_score"] = df["revenue_efficiency"].map(map_score)
    df["payback_score"] = df["payback_health"].map(map_score)

    df["growth_health_score"] = (
        df["traffic_score"] * 0.10 +
        df["activation_score"] * 0.15 +
        df["early_signal_score_norm"] * 0.20 +
        df["retention_score"] * 0.20 +
        df["revenue_score"] * 0.20 +
        df["payback_score"] * 0.15
    ).round(1)
    df["growth_health_category"] = df["growth_health_score"].apply(score_category)

    # Measurement
    df["os_tracking_risk"] = df["os"].apply(os_tracking_risk)

    def measurement_confidence_score(row):
        score = 100
        if pd.isna(row["spend"]) or row["spend"] == 0:
            score -= 50
        if pd.isna(row["installs"]) or row["installs"] == 0:
            score -= 30
        if str(row["os"]).strip().lower() == "ios":
            score -= 15
        if row["skan_only"]:
            score -= 20
        return max(score, 0)

    df["measurement_confidence_score"] = df.apply(measurement_confidence_score, axis=1)

    def measurement_confidence_level(score):
        if pd.isna(score):
            return "Unknown"
        if score >= 80:
            return "High"
        if score >= 60:
            return "Medium"
        if score >= 40:
            return "Low"
        return "Very Low"

    df["measurement_confidence_level"] = df["measurement_confidence_score"].apply(measurement_confidence_level)

    def measurement_flag(score):
        if pd.isna(score):
            return "Unknown"
        if score < 40:
            return "Unreliable Data"
        if score < 60:
            return "Directional Only"
        return "Reliable"

    df["measurement_flag"] = df["measurement_confidence_score"].apply(measurement_flag)

    def final_recommendation_v4(row):
        if row["measurement_confidence_score"] < 40 and row["growth_health_score"] < 60:
            return "Measurement Test Required"
        if row["strategic_channel"] and row["measurement_confidence_score"] < 50:
            return "Strategic Keep"
        if row["growth_health_score"] >= 75 and row["measurement_confidence_score"] >= 60:
            return "Scale"
        if row["growth_health_score"] >= 60 and row["measurement_confidence_score"] >= 40:
            return "Maintain"
        if row["growth_health_score"] < 50 and row["measurement_confidence_score"] >= 60:
            return "Reduce"
        if row["allocation_suggestion"] == "Optimize":
            return "Optimize"
        return row["allocation_suggestion"]

    df["final_recommendation_v4"] = df.apply(final_recommendation_v4, axis=1)

    def build_audit_summary_v4(row):
        return (
            f"Bottleneck: {row['primary_bottleneck']} | "
            f"Growth score: {row['growth_health_score']} | "
            f"Measurement: {row['measurement_confidence_level']} | "
            f"Action: {row['final_recommendation_v4']}"
        )

    df["audit_summary"] = df.apply(build_audit_summary_v4, axis=1)

    audit_df = df.copy()

    summary = {
        "total_spend": df["spend"].sum(skipna=True),
        "total_installs": df["installs"].sum(skipna=True),
        "avg_growth_health_score": df["growth_health_score"].mean(skipna=True),
        "avg_measurement_confidence_score": df["measurement_confidence_score"].mean(skipna=True),
        "high_risk_count": (df["growth_risk_level"] == "High").sum(),
        "scale_count": (df["final_recommendation_v4"] == "Scale").sum(),
        "reduce_count": (df["final_recommendation_v4"] == "Reduce").sum(),
        "measurement_test_required_count": (df["final_recommendation_v4"] == "Measurement Test Required").sum(),
        "strategic_keep_count": (df["final_recommendation_v4"] == "Strategic Keep").sum(),
    }
    summary_df = pd.DataFrame([summary])

    channel_summary = df.groupby("channel").agg(
        spend=("spend", "sum"),
        installs=("installs", "sum"),
        revenue=("revenue", "sum"),
        avg_growth_health_score=("growth_health_score", "mean"),
        avg_measurement_confidence_score=("measurement_confidence_score", "mean"),
        avg_payback=("payback_period", "mean"),
    ).reset_index()

    bottleneck_counts = df["primary_bottleneck"].value_counts().reset_index()
    bottleneck_counts.columns = ["primary_bottleneck", "count"]

    final_reco_counts = df["final_recommendation_v4"].value_counts().reset_index()
    final_reco_counts.columns = ["final_recommendation_v4", "count"]

    return audit_df, summary_df, channel_summary, bottleneck_counts, final_reco_counts

# --------------------------------------------------
# Table styling
# --------------------------------------------------
def highlight_growth_score(val):
    if pd.isna(val):
        return ""
    if val < 50:
        return "background-color: rgba(239, 68, 68, 0.35); color: white;"
    if val < 60:
        return "background-color: rgba(245, 158, 11, 0.30); color: white;"
    return ""

def highlight_measurement_score(val):
    if pd.isna(val):
        return ""
    if val < 40:
        return "background-color: rgba(220, 38, 38, 0.38); color: white;"
    if val < 60:
        return "background-color: rgba(234, 179, 8, 0.30); color: white;"
    return ""

# --------------------------------------------------
# App header
# --------------------------------------------------
st.markdown("## AI-assisted Growth Audit System")
st.markdown(
    "<div class='small-note'>Performance × Measurement Confidence × Strategic Decision</div>",
    unsafe_allow_html=True
)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.markdown("## Upload")

st.sidebar.markdown("### Required columns")
st.sidebar.markdown(
    f"""
    <div class="mono-box">
    {"<br>".join(REQUIRED_COLUMNS)}
    </div>
    """,
    unsafe_allow_html=True
)
st.sidebar.caption("CSV must include all required columns exactly as shown above.")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is None:
    st.info("CSV 파일을 업로드하면 Executive Dashboard가 생성됩니다.")
else:
    try:
        raw_df = pd.read_csv(uploaded_file)
        audit_df, summary_df, channel_summary, bottleneck_counts, final_reco_counts = run_growth_audit_v4(raw_df.copy())

        # Executive KPIs
        st.markdown("### Executive Overview")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Avg Growth Score", f"{summary_df.loc[0, 'avg_growth_health_score']:.1f}")
        c2.metric("Avg Measurement Confidence", f"{summary_df.loc[0, 'avg_measurement_confidence_score']:.1f}")
        c3.metric("High Risk", int(summary_df.loc[0, "high_risk_count"]))
        c4.metric("Scale", int(summary_df.loc[0, "scale_count"]))
        c5.metric("Measurement Test", int(summary_df.loc[0, "measurement_test_required_count"]))

        # Management summary
        st.markdown("### Management Summary")
        top_bottleneck = bottleneck_counts.iloc[0]["primary_bottleneck"] if len(bottleneck_counts) > 0 else "N/A"
        top_bottleneck_count = bottleneck_counts.iloc[0]["count"] if len(bottleneck_counts) > 0 else 0

        st.markdown(f"""
- **Average Growth Health Score:** {summary_df.loc[0, 'avg_growth_health_score']:.1f}  
- **Average Measurement Confidence:** {summary_df.loc[0, 'avg_measurement_confidence_score']:.1f}  
- **Most common bottleneck:** `{top_bottleneck}` ({top_bottleneck_count} campaigns)  
- **Scale candidates:** {int(summary_df.loc[0, 'scale_count'])}  
- **Measurement test required:** {int(summary_df.loc[0, 'measurement_test_required_count'])}  
- **Strategic keep:** {int(summary_df.loc[0, 'strategic_keep_count'])}
""")
        
        # Filters
        st.markdown("### Campaign Explorer")
        f1, f2, f3, f4 = st.columns(4)

        channel_options = ["All"] + sorted(audit_df["channel"].dropna().unique().tolist())
        os_options = ["All"] + sorted(audit_df["os"].dropna().unique().tolist())
        reco_options = ["All"] + sorted(audit_df["final_recommendation_v4"].dropna().unique().tolist())
        bottleneck_options = ["All"] + sorted(audit_df["primary_bottleneck"].dropna().unique().tolist())

        selected_channel = f1.selectbox("Channel", channel_options)
        selected_os = f2.selectbox("OS", os_options)
        selected_reco = f3.selectbox("Final Recommendation", reco_options)
        selected_bottleneck = f4.selectbox("Bottleneck", bottleneck_options)

        filtered_df = audit_df.copy()

        if selected_channel != "All":
            filtered_df = filtered_df[filtered_df["channel"] == selected_channel]
        if selected_os != "All":
            filtered_df = filtered_df[filtered_df["os"] == selected_os]
        if selected_reco != "All":
            filtered_df = filtered_df[filtered_df["final_recommendation_v4"] == selected_reco]
        if selected_bottleneck != "All":
            filtered_df = filtered_df[filtered_df["primary_bottleneck"] == selected_bottleneck]

        # Charts row 1
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Bottleneck Distribution")
            st.bar_chart(bottleneck_counts.set_index("primary_bottleneck"))

        with col2:
            st.markdown("### Final Recommendation Distribution")
            st.bar_chart(final_reco_counts.set_index("final_recommendation_v4"))

        # Scatter plot
        st.markdown("### Strategic Scatter Plot")
        scatter_df = filtered_df.copy()
        scatter_df["campaign_label"] = scatter_df["channel"].astype(str) + " | " + scatter_df["campaign"].astype(str)

        scatter = (
            alt.Chart(scatter_df)
            .mark_circle(size=140)
            .encode(
                x=alt.X("growth_health_score:Q", title="Growth Health Score"),
                y=alt.Y("measurement_confidence_score:Q", title="Measurement Confidence Score"),
                color=alt.Color("final_recommendation_v4:N", title="Final Recommendation"),
                tooltip=[
                    alt.Tooltip("channel:N", title="Channel"),
                    alt.Tooltip("campaign:N", title="Campaign"),
                    alt.Tooltip("os:N", title="OS"),
                    alt.Tooltip("growth_health_score:Q", title="Growth Score", format=".1f"),
                    alt.Tooltip("measurement_confidence_score:Q", title="Measurement Score", format=".1f"),
                    alt.Tooltip("primary_bottleneck:N", title="Bottleneck"),
                    alt.Tooltip("final_recommendation_v4:N", title="Recommendation"),
                ],
            )
            .properties(height=420)
            .interactive()
        )
        st.altair_chart(scatter, use_container_width=True)

        # Channel summary
        st.markdown("### Channel Intelligence")
        st.dataframe(channel_summary, use_container_width=True)

        # Styled campaign audit table
        st.markdown("### Campaign Audit Table")
        view_columns = [
            "channel", "campaign", "os",
            "growth_health_score", "measurement_confidence_score",
            "primary_bottleneck", "final_recommendation_v4", "audit_summary"
        ]

        styled_df = (
            filtered_df[view_columns]
            .style
            .applymap(highlight_growth_score, subset=["growth_health_score"])
            .applymap(highlight_measurement_score, subset=["measurement_confidence_score"])
        )

        st.dataframe(styled_df, use_container_width=True, height=420)


# --------------------------------------------------
        # AI Operational Insight (日本語 Playbook)
        # --------------------------------------------------
        st.markdown("### 🤖 AI 戦略・運用インサイト")
        
        # フィルタリングされたデータの中から、最も頻度の高いボトルネックを特定
        if not filtered_df.empty:
            main_bottleneck = filtered_df["primary_bottleneck"].mode()[0]
            
            # 日本語版マーケティング運用プレイブックの定義
            playbook_content = {
                "Weak Day-1 fit": {
                    "title": "流入とプロダクトの不一致 (Creative-Targeting Misalignment)",
                    "desc": "広告で期待させた作品や体験が、アプリ起動直後に提供されていません。ユーザーの期待値とのギャップが生じています。",
                    "actions": [
                        "広告クリエイティブに使用した作品を、アプリのホーム上部バナーにも固定表示し、導線を一致させる",
                        "継続率の高い特定ジャンル（ロマンスファンタジー等）以外のターゲティングを一時縮小",
                        "クリエイティブ内に『待てば無料』システムの説明を加え、ルールを理解した高関心層のみを誘導"
                    ]
                },
                "Early activation decay": {
                    "title": "初期アクティベーションの失敗 (Early Value Delivery Issue)",
                    "desc": "インストール後、最初の1話閲覧や無料分消化までに離脱が発生しています。",
                    "actions": [
                        "広告クリック時、作品詳細ページではなく『第1話リスト』へ直接遷移するディープリンクの動作確認と最適化",
                        "演出重視の素材よりも、『1話無料』『期間限定チケット配布』など即時的なインセンティブを強調",
                        "インストール後1時間以内に未実行のユーザーに対し、CRMプッシュ通知やリマーケティングを集中投下"
                    ]
                },
                "Monetization weakness": {
                    "title": "収益化のボトルネック (Purchase Conversion Barrier)",
                    "desc": "ユーザーは残存していますが、コイン購入や課金ページでの離脱が目立ちます。",
                    "actions": [
                        "インストール最適化ではなく、課金完了（Purchase）イベントを最適化基準とするキャンペーンを強化",
                        "ARPPUが検証済みの高価値チャネル（ASAのキーワード広告等）へ予算をシフト",
                        "クリエイティブ内で『初回購入特典』や『コイン還元キャンペーン』を直接露出し、購買意欲の高い層をフィルタリング"
                    ]
                },
                "Structural retention decay": {
                    "title": "長期継続率の不足 (Long-term Retention Risk)",
                    "desc": "短期的な体験だけで満足し、アプリに定着する動機付けが不足しています。",
                    "actions": [
                        "読み切り作品の素材を減らし、200話以上の『長期連載作品』の素材比率を拡大",
                        "媒体側の『7日後再訪問ユーザー』最適化ビッディング（AC 2.0/3.0等）を導入",
                        "お気に入り登録の誘導など、既存のCRMシナリオと連動したリテンション広告の展開"
                    ]
                }
            }

            # 該当するボトルネックに応じたガイドを表示
            guide = playbook_content.get(main_bottleneck, {
                "title": "総合効率の最適化 (General Optimization)",
                "desc": "特定のボトルネックではなく、全体的な指標管理が必要なフェーズです。",
                "actions": ["高効率キャンペーン（Scale）の予算増額", "低効率媒体の予算削減とASAへの予算シフト", "データ信頼性（Confidence）の再検証"]
            })

            # UI レンダリング (HTML/CSS)
            st.markdown(f"""
            <div class="mono-box">
                <div style="color: #60a5fa; font-size: 1.1rem; font-weight: 700;">🎯 重点改善タスク: {guide['title']}</div>
                <div style="color: #9ca3af; margin-bottom: 10px;">現状分析: {guide['desc']}</div>
                <div style="margin-left: 10px;">
                    {"".join([f"<div style='margin-bottom: 5px;'>• {a}</div>" for a in guide['actions']])}
                </div>
                <div style="margin-top: 10px; font-size: 0.85rem; color: #f87171;">
                    ⚠️ この提案は、開発・デザインの修正を行わず、<b>マーケティング運用および予算配分の最適化</b>のみで即座に実行可能です。
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("分析対象のデータがありません。")
            
        
        # Download Center
        st.markdown("### Download Center")
        d1, d2 = st.columns(2)

        d1.download_button(
            "Download Audit Output CSV",
            audit_df.to_csv(index=False).encode("utf-8"),
            file_name="growth_audit_output_v4.csv",
            mime="text/csv",
        )

        d2.download_button(
            "Download Channel Summary CSV",
            channel_summary.to_csv(index=False).encode("utf-8"),
            file_name="growth_channel_summary_v4.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"Error: {e}")
