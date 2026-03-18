import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(
    page_title="AI 成長監査システム (Growth Audit System) v0.4",
    layout="wide"
)

# --------------------------------------------------
# カスタムスタイル
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
# 必須カラム定義
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
# ヘルパー関数
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
        return "不明"
    if value <= avg_value * 0.85:
        return "良好"
    elif value <= avg_value * 1.15:
        return "普通"
    return "注意"

def rate_relative_high_is_good(value, avg_value):
    if pd.isna(value) or pd.isna(avg_value):
        return "不明"
    if value >= avg_value * 1.15:
        return "良好"
    elif value >= avg_value * 0.85:
        return "普通"
    return "注意"

def os_tracking_risk(os_value):
    if str(os_value).strip().lower() == "ios":
        return "ATT リスク"
    if str(os_value).strip().lower() == "android":
        return "低リスク"
    return "不明"

def map_score(value):
    score_map = {
        "良好": 100,
        "普通": 60,
        "注意": 30,
        "リスクあり": 20,
        "不明": 50,
    }
    return score_map.get(value, 50)

def score_category(score):
    if pd.isna(score):
        return "不明"
    if score >= 80:
        return "健全な成長"
    if score >= 60:
        return "最適化の余地あり"
    if score >= 40:
        return "構造的な成長課題"
    return "致命的な成長リスク"

# --------------------------------------------------
# コアエンジン
# --------------------------------------------------
def run_growth_audit_v4(df: pd.DataFrame):
    required_columns = REQUIRED_COLUMNS

    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"必須カラムが不足しています: {missing_cols}")

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

    # コア指標の計算
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
            return "不明"
        if activation_rate >= 0.50:
            return "良好"
        elif activation_rate >= 0.30:
            return "普通"
        return "注意"

    def score_retention_stability(d7_retention):
        if pd.isna(d7_retention):
            return "不明"
        if d7_retention >= 0.25:
            return "良好"
        elif d7_retention >= 0.15:
            return "普通"
        return "注意"

    def score_early_signal(early_signal_score):
        if pd.isna(early_signal_score):
            return "不明"
        if early_signal_score >= 30:
            return "良好"
        elif early_signal_score >= 18:
            return "普通"
        return "注意"

    def score_curve_quality(curve_quality):
        if pd.isna(curve_quality):
            return "不明"
        if curve_quality >= 0.45:
            return "良好"
        elif curve_quality >= 0.30:
            return "普通"
        return "注意"

    def score_revenue_efficiency(arpu):
        return rate_relative_high_is_good(arpu, avg_arpu)

    def score_payback_health(payback_period):
        if pd.isna(payback_period):
            return "不明"
        if payback_period <= 1.5:
            return "良好"
        elif payback_period <= 3.0:
            return "普通"
        return "リスクあり"

    df["traffic_efficiency"] = df["cpi"].apply(score_traffic_efficiency)
    df["activation_efficiency"] = df["activation_rate"].apply(score_activation_efficiency)
    df["retention_stability"] = df["d7_retention"].apply(score_retention_stability)
    df["early_signal_health"] = df["early_signal_score"].apply(score_early_signal)
    df["curve_quality_health"] = df["retention_curve_quality"].apply(score_curve_quality)
    df["revenue_efficiency"] = df["arpu"].apply(score_revenue_efficiency)
    df["payback_health"] = df["payback_period"].apply(score_payback_health)

    def detect_primary_bottleneck_v2(row):
        if row["d1_retention"] < 0.20:
            return "Day-1 適合性の不足"
        if row["d1_to_d3_drop"] > 0.20:
            return "初期アクティベーションの離脱"
        if row["d3_to_d7_drop"] > 0.15:
            return "構造的な継続率の低下"
        if row["activation_efficiency"] == "注意":
            return "アクティベーションの摩擦"
        if row["revenue_efficiency"] == "注意" and row["retention_stability"] in ["普通", "良好"]:
            return "収益化の弱点"
        if row["payback_health"] == "リスクあり":
            return "資本効率の悪化"
        if row["traffic_efficiency"] == "注意" and row["retention_stability"] == "注意":
            return "獲得効率の低下"
        return "重大なボトルネックなし"

    df["primary_bottleneck"] = df.apply(detect_primary_bottleneck_v2, axis=1)

    def growth_risk_level_v2(row):
        weak_signals = 0
        diagnostic_fields = [
            row["traffic_efficiency"], row["activation_efficiency"],
            row["retention_stability"], row["early_signal_health"],
            row["curve_quality_health"], row["revenue_efficiency"],
        ]
        weak_signals += sum(1 for x in diagnostic_fields if x == "注意")
        weak_signals += 1 if row["payback_health"] == "リスクあり" else 0

        if row["primary_bottleneck"] in [
            "Day-1 適合性の不足", "初期アクティベーションの離脱", "構造的な継続率の低下"
        ] and row["payback_health"] == "リスクあり":
            return "高"
        if weak_signals >= 3:
            return "高"
        if weak_signals >= 1:
            return "中"
        return "低"

    df["growth_risk_level"] = df.apply(growth_risk_level_v2, axis=1)

    def allocation_suggestion_v2(row):
        if (
            row["early_signal_health"] == "良好"
            and row["payback_health"] == "良好"
            and row["revenue_efficiency"] in ["普通", "良好"]
        ):
            return "拡大 (Scale)"
        if row["primary_bottleneck"] in [
            "Day-1 適合性の不足", "初期アクティベーションの離脱", "アクティベーションの摩擦"
        ]:
            return "最適化 (Optimize)"
        if row["primary_bottleneck"] in [
            "構造的な継続率の低下", "収益化の弱点"
        ] and row["payback_health"] != "リスクあり":
            return "維持 (Maintain)"
        if row["payback_health"] == "リスクあり" and row["growth_risk_level"] in ["中", "高"]:
            return "縮小 (Reduce)"
        return "維持 (Maintain)"

    df["allocation_suggestion"] = df.apply(allocation_suggestion_v2, axis=1)

    # 成長ヘルスのスコアリング
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

    # 計測の信頼性
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
            return "不明"
        if score >= 80:
            return "高"
        if score >= 60:
            return "中"
        if score >= 40:
            return "低"
        return "極めて低"

    df["measurement_confidence_level"] = df["measurement_confidence_score"].apply(measurement_confidence_level)

    def measurement_flag(score):
        if pd.isna(score):
            return "不明"
        if score < 40:
            return "信頼性なし"
        if score < 60:
            return "参考程度"
        return "信頼可能"

    df["measurement_flag"] = df["measurement_confidence_score"].apply(measurement_flag)

    def final_recommendation_v4(row):
        if row["measurement_confidence_score"] < 40 and row["growth_health_score"] < 60:
            return "計測テストが必要"
        if row["strategic_channel"] and row["measurement_confidence_score"] < 50:
            return "戦略的維持"
        if row["growth_health_score"] >= 75 and row["measurement_confidence_score"] >= 60:
            return "拡大 (Scale)"
        if row["growth_health_score"] >= 60 and row["measurement_confidence_score"] >= 40:
            return "維持 (Maintain)"
        if row["growth_health_score"] < 50 and row["measurement_confidence_score"] >= 60:
            return "縮小 (Reduce)"
        if row["allocation_suggestion"] == "最適化 (Optimize)":
            return "最適化 (Optimize)"
        return row["allocation_suggestion"]

    df["final_recommendation_v4"] = df.apply(final_recommendation_v4, axis=1)

    def build_audit_summary_v4(row):
        return (
            f"ボトルネック: {row['primary_bottleneck']} | "
            f"成長スコア: {row['growth_health_score']} | "
            f"計測信頼性: {row['measurement_confidence_level']} | "
            f"アクション: {row['final_recommendation_v4']}"
        )

    df["audit_summary"] = df.apply(build_audit_summary_v4, axis=1)

    audit_df = df.copy()

    summary = {
        "total_spend": df["spend"].sum(skipna=True),
        "total_installs": df["installs"].sum(skipna=True),
        "avg_growth_health_score": df["growth_health_score"].mean(skipna=True),
        "avg_measurement_confidence_score": df["measurement_confidence_score"].mean(skipna=True),
        "high_risk_count": (df["growth_risk_level"] == "高").sum(),
        "scale_count": (df["final_recommendation_v4"] == "拡大 (Scale)").sum(),
        "reduce_count": (df["final_recommendation_v4"] == "縮小 (Reduce)").sum(),
        "measurement_test_required_count": (df["final_recommendation_v4"] == "計測テストが必要").sum(),
        "strategic_keep_count": (df["final_recommendation_v4"] == "戦略적維持").sum(),
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
# テーブルスタイリング
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
# アプリヘッダー
# --------------------------------------------------
st.markdown("## AI 成長監査システム (Growth Audit System)")
st.markdown(
    "<div class='small-note'>パフォーマンス × 計測信頼性 × 戦略的決定</div>",
    unsafe_allow_html=True
)

# --------------------------------------------------
# サイドバー
# --------------------------------------------------
st.sidebar.markdown("## アップロード")

st.sidebar.markdown("### 必須カラム")
st.sidebar.markdown(
    f"""
    <div class="mono-box">
    {"<br>".join(REQUIRED_COLUMNS)}
    </div>
    """,
    unsafe_allow_html=True
)
st.sidebar.caption("CSVには上記の必須カラムが正確に含まれている必要があります。")

uploaded_file = st.sidebar.file_uploader("CSVをアップロード", type=["csv"])

if uploaded_file is None:
    st.info("CSVファイルをアップロードすると、エグゼクティブ・ダッシュボードが生成されます。")
else:
    try:
        raw_df = pd.read_csv(uploaded_file)
        audit_df, summary_df, channel_summary, bottleneck_counts, final_reco_counts = run_growth_audit_v4(raw_df.copy())

        # エグゼクティブ KPI
        st.markdown("### エグゼクティブ・オーバービュー")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("平均成長スコア", f"{summary_df.loc[0, 'avg_growth_health_score']:.1f}")
        c2.metric("平均計測信頼性", f"{summary_df.loc[0, 'avg_measurement_confidence_score']:.1f}")
        c3.metric("高リスク件数", int(summary_df.loc[0, "high_risk_count"]))
        c4.metric("拡大推奨", int(summary_df.loc[0, "scale_count"]))
        c5.metric("計測テスト対象", int(summary_df.loc[0, "measurement_test_required_count"]))

        # 経営サマリー
        st.markdown("### 経営サマリー")
        top_bottleneck = bottleneck_counts.iloc[0]["primary_bottleneck"] if len(bottleneck_counts) > 0 else "N/A"
        top_bottleneck_count = bottleneck_counts.iloc[0]["count"] if len(bottleneck_counts) > 0 else 0

        st.markdown(f"""
- **平均成長ヘルススコア:** {summary_df.loc[0, 'avg_growth_health_score']:.1f}  
- **平均計測信頼性スコア:** {summary_df.loc[0, 'avg_measurement_confidence_score']:.1f}  
- **最も一般的なボトルネック:** `{top_bottleneck}` ({top_bottleneck_count} キャンペーン)  
- **拡大候補数:** {int(summary_df.loc[0, 'scale_count'])}  
- **計測テストが必要なキャンペーン:** {int(summary_df.loc[0, 'measurement_test_required_count'])}  
- **戦略的維持キャンペーン:** {int(summary_df.loc[0, 'strategic_keep_count'])}
""")
        
        # フィルター
        st.markdown("### キャンペーン・エクスプローラー")
        f1, f2, f3, f4, f5 = st.columns(5) # 컬럼을 5개로 확장

        channel_options = ["すべて"] + sorted(audit_df["channel"].dropna().unique().tolist())
        campaign_options = ["すべて"] + sorted(audit_df["campaign"].dropna().unique().tolist()) # 캠페인 옵션 추가
        os_options = ["すべて"] + sorted(audit_df["os"].dropna().unique().tolist())
        reco_options = ["すべて"] + sorted(audit_df["final_recommendation_v4"].dropna().unique().tolist())
        bottleneck_options = ["すべて"] + sorted(audit_df["primary_bottleneck"].dropna().unique().tolist())

        selected_channel = f1.selectbox("チャネル", channel_options)
        selected_campaign = f2.selectbox("キャンペーン", campaign_options) # 캠페인 선택창 추가
        selected_os = f3.selectbox("OS", os_options)
        selected_reco = f4.selectbox("最終推奨アクション", reco_options)
        selected_bottleneck = f5.selectbox("ボトルネック", bottleneck_options)

        filtered_df = audit_df.copy()

        # 필터 적용 로직
        if selected_channel != "すべて":
            filtered_df = filtered_df[filtered_df["channel"] == selected_channel]
        if selected_campaign != "すべて": # 캠페인 필터 적용
            filtered_df = filtered_df[filtered_df["campaign"] == selected_campaign]
        if selected_os != "すべて":
            filtered_df = filtered_df[filtered_df["os"] == selected_os]
        if selected_reco != "すべて":
            filtered_df = filtered_df[filtered_df["final_recommendation_v4"] == selected_reco]
        if selected_bottleneck != "すべて":
            filtered_df = filtered_df[filtered_df["primary_bottleneck"] == selected_bottleneck]
 
        # チャート 1行目
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### ボトルネック分布")
            st.bar_chart(bottleneck_counts.set_index("primary_bottleneck"))

        with col2:
            st.markdown("### 最終推奨アクションの分布")
            st.bar_chart(final_reco_counts.set_index("final_recommendation_v4"))

        # 散布図
        st.markdown("### 戦略的散布図 (Strategic Scatter Plot)")
        scatter_df = filtered_df.copy()
        scatter_df["campaign_label"] = scatter_df["channel"].astype(str) + " | " + scatter_df["campaign"].astype(str)

        scatter = (
            alt.Chart(scatter_df)
            .mark_circle(size=140)
            .encode(
                x=alt.X("growth_health_score:Q", title="成長ヘルススコア"),
                y=alt.Y("measurement_confidence_score:Q", title="計測信頼性スコア"),
                color=alt.Color("final_recommendation_v4:N", title="最終推奨"),
                tooltip=[
                    alt.Tooltip("channel:N", title="チャネル"),
                    alt.Tooltip("campaign:N", title="キャンペーン"),
                    alt.Tooltip("os:N", title="OS"),
                    alt.Tooltip("growth_health_score:Q", title="成長スコア", format=".1f"),
                    alt.Tooltip("measurement_confidence_score:Q", title="計測スコア", format=".1f"),
                    alt.Tooltip("primary_bottleneck:N", title="ボトルネック"),
                    alt.Tooltip("final_recommendation_v4:N", title="推奨アクション"),
                ],
            )
            .properties(height=420)
            .interactive()
        )
        st.altair_chart(scatter, use_container_width=True)


# AI 戦略・運用インサイト (Playbook) - 정교화 버전
        st.markdown("### 🤖 AI 戦略・運用インサイト (Advanced Playbook)")
        
        if not filtered_df.empty:
            # 1. 동적 데이터 추출
            main_bottleneck = filtered_df["primary_bottleneck"].mode()[0]
            avg_health = filtered_df["growth_health_score"].mean()
            ios_ratio = (filtered_df["os"].str.lower() == "ios").mean()
            low_confidence_count = (filtered_df["measurement_confidence_level"].isin(["低", "極めて低"])).sum()
            
            # 2. 동적 진단 메시지 생성 (선택한 캠페인 이름 반영)
            if selected_campaign != "すべて":
                target_name = f"キャンペーン「{selected_campaign}」"
            else:
                target_name = "現在表示中のデータ"
                
            diagnostic_msg = f"{target_name}の主たるボトルネックは「<b>{main_bottleneck}</b>」です。"
            
            if avg_health < 50:
                diagnostic_msg += f"<br>⚠️ 平均成長スコアが <b>{avg_health:.1f}</b> と低迷しており、早急な予算リアロケーションが推奨されます。"
            elif avg_health >= 70:
                diagnostic_msg += f"<br>✅ 全体的に健全なスコア（<b>{avg_health:.1f}</b>）を維持しています。さらなるスケールアップの機会を探りましょう。"

            # 3. 병목별 핵심 전략 프레임워크 (Immediate, Optimize, Measurement 3단계)
            playbook_content = {
                "Day-1 適合性の不足": {
                    "title": "流入とプロダクトの不一致 (Creative-Targeting Misalignment)",
                    "immediate": ["Day-1継続率が20%未満のキャンペーンは直ちに日予算を50%削減または停止"],
                    "optimize": [
                        "広告クリエイティブに使用したフック（絵柄やオファー）とアプリ起動直後の画面を一致させる",
                        "継続率の高い特定ジャンル以外のターゲティングを一時縮小"
                    ]
                },
                "初期アクティベーションの離脱": {
                    "title": "初期アクティベーションの失敗 (Early Value Delivery Issue)",
                    "immediate": ["アクティベーション単価（CPA）が許容範囲を30%超えるセグメントの入札額を下げる"],
                    "optimize": [
                        "クリック時のディープリンク先を最適化し、ユーザーに最短でコアバリュー（第1話無料など）を体験させる",
                        "インストール後24時間以内のプッシュ通知/CRMシナリオを強化"
                    ]
                },
                "収益化の弱点": {
                    "title": "収益化のボトルネック (Purchase Conversion Barrier)",
                    "immediate": ["インストール目的ではなく、課金（Purchase）イベント最適化キャンペーンへ予算をシフト"],
                    "optimize": [
                        "初回課金ハードルを下げる限定オファー（初回コイン増量など）をクリエイティブで直接訴求",
                        "LTVの高い既存ユーザーの類似オーディエンス（Lookalike）を活用"
                    ]
                },
                "構造的な継続率の低下": {
                    "title": "長期継続率の不足 (Long-term Retention Risk)",
                    "immediate": ["ROAS回収期間（Payback Period）が長期化している媒体の目標CPAを厳格化"],
                    "optimize": [
                        "読み切り作品より、長期連載・課金単価の高いコンテンツにフォーカスした広告素材の投下",
                        "媒体側の『7日後再訪問ユーザー』最適化（AEO）を導入"
                    ]
                }
            }

            # 기본값(Fallback)
            guide = playbook_content.get(main_bottleneck, {
                "title": "総合効率の最適化 (General Optimization)",
                "immediate": ["低効率キャンペーンから高効率キャンペーンへの予算シフト（上位20%へ集中投資）"],
                "optimize": ["クリエイティブの摩耗（Ad Fatigue）チェックと新規素材のテスト"]
            })

            # 4. OS 및 측정 환경에 따른 동적 액션 추가 (Contextual Actions)
            measurement_actions = []
            if ios_ratio > 0.4:
                measurement_actions.append("🍏 <b>iOS比率が高い環境:</b> SKAdNetworkのコンバージョン値マッピングが現在のボトルネック（例: 課金/継続）を適切にキャッチできているか再設計を推奨。")
            if low_confidence_count > 0:
                measurement_actions.append(f"🔍 <b>計測リスク警告:</b> {low_confidence_count}件のキャンペーンで計測信頼性が低いため、MMM（マーケティング・ミックス・モデリング）やインクリメンタリティ・テストでの真の成果検証が必要です。")
            
            if not measurement_actions:
                measurement_actions.append("📊 計測環境は概ね健全ですが、イベント欠損がないか定期的にモニタリングしてください。")

            # 5. UI 렌더링 (모든 내부 들여쓰기 완벽 제거)
            immediate_html = "".join([f"<div style='font-size: 0.9rem; margin-bottom: 4px;'>• {a}</div>" for a in guide['immediate']])
            optimize_html = "".join([f"<div style='font-size: 0.9rem; margin-bottom: 4px;'>• {a}</div>" for a in guide['optimize']])
            measurement_html = "".join([f"<div style='font-size: 0.9rem; margin-bottom: 4px;'>{a}</div>" for a in measurement_actions])

            html_content = f"""
<div class="info-card">
<div style="color: #60a5fa; font-size: 1.15rem; font-weight: 700; margin-bottom: 0.5rem;">
🎯 重点改善テーマ: {guide['title']}
</div>
<div style="color: #d1d5db; margin-bottom: 1.2rem; font-size: 0.95rem;">
{diagnostic_msg}
</div>
<div style="display: flex; flex-direction: column; gap: 1rem;">
<div style="background-color: rgba(239, 68, 68, 0.1); border-left: 4px solid #ef4444; padding: 0.8rem; border-radius: 4px;">
<div style="color: #ef4444; font-weight: bold; margin-bottom: 0.4rem;">🚨 応急処置 (即時リアロケーション)</div>
{immediate_html}
</div>
<div style="background-color: rgba(59, 130, 246, 0.1); border-left: 4px solid #3b82f6; padding: 0.8rem; border-radius: 4px;">
<div style="color: #3b82f6; font-weight: bold; margin-bottom: 0.4rem;">🛠️ キャンペーン・クリエイティブ最適化</div>
{optimize_html}
</div>
<div style="background-color: rgba(245, 158, 11, 0.1); border-left: 4px solid #f59e0b; padding: 0.8rem; border-radius: 4px;">
<div style="color: #f59e0b; font-weight: bold; margin-bottom: 0.4rem;">📊 データ・計測環境の検証</div>
{measurement_html}
</div>
</div>
</div>
"""
            st.markdown(html_content, unsafe_allow_html=True)
            
        else:
            st.warning("分析対象のデータがありません。フィルター条件を変更してください。")
        
        
        # チャネル・サ마リー
        st.markdown("### チャネル・インテリジェンス (Channel Intelligence)")
        st.dataframe(channel_summary, use_container_width=True)

        # 監査テーブル
        st.markdown("### キャンペーン監査テーブル (Campaign Audit Table)")
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

        # ダウンロード・センター
        st.markdown("### ダウンロード・センター (Download Center)")
        d1, d2 = st.columns(2)

        d1.download_button(
            "監査出力 (CSV) をダウンロード",
            audit_df.to_csv(index=False).encode("utf-8"),
            file_name="growth_audit_output_jp.csv",
            mime="text/csv",
        )

        d2.download_button(
            "チャネルサマリー (CSV) をダウンロード",
            channel_summary.to_csv(index=False).encode("utf-8"),
            file_name="growth_channel_summary_jp.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"エラーが発生しました: {e}")
