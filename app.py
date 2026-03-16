import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="AI-assisted Growth Audit System v0.4",
    layout="wide"
)

st.title("AI-assisted Growth Audit System v0.4")
st.caption("Performance × Measurement Confidence × Strategic Decision")

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
    required_columns = [
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

    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    numeric_cols = [
        "spend",
        "installs",
        "activated_users",
        "d1_retention",
        "d3_retention",
        "d7_retention",
        "revenue",
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

    def predict_payback_v2(row):
        if pd.isna(row["cpi"]) or pd.isna(row["arpu"]) or row["arpu"] <= 0:
            return np.nan
        if pd.isna(row["predicted_ltv"]) or pd.isna(row["estimated_ltv"]) or row["estimated_ltv"] <= 0:
            return np.nan

        uplift_ratio = row["predicted_ltv"] / row["estimated_ltv"]
        return row["cpi"] / (row["arpu"] * uplift_ratio)

    df["predicted_payback"] = df.apply(predict_payback_v2, axis=1)

    # Benchmarks
    avg_cpi = df["cpi"].mean(skipna=True)
    avg_arpu = df["arpu"].mean(skipna=True)

    # Diagnostic scoring
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

    # Bottleneck detection
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

    # Growth risk
    def growth_risk_level_v2(row):
        weak_signals = 0

        diagnostic_fields = [
            row["traffic_efficiency"],
            row["activation_efficiency"],
            row["retention_stability"],
            row["early_signal_health"],
            row["curve_quality_health"],
            row["revenue_efficiency"],
        ]

        weak_signals += sum(1 for x in diagnostic_fields if x == "Weak")
        weak_signals += 1 if row["payback_health"] == "Risky" else 0

        if row["primary_bottleneck"] in [
            "Weak Day-1 fit",
            "Early activation decay",
            "Structural retention decay",
        ] and row["payback_health"] == "Risky":
            return "High"

        if weak_signals >= 3:
            return "High"

        if weak_signals >= 1:
            return "Medium"

        return "Low"

    df["growth_risk_level"] = df.apply(growth_risk_level_v2, axis=1)

    # Performance-only suggestion
    def allocation_suggestion_v2(row):
        if (
            row["early_signal_health"] == "Strong"
            and row["payback_health"] == "Strong"
            and row["revenue_efficiency"] in ["Moderate", "Strong"]
        ):
            return "Scale"

        if row["primary_bottleneck"] in [
            "Weak Day-1 fit",
            "Early activation decay",
            "Activation friction",
        ]:
            return "Optimize"

        if row["primary_bottleneck"] in [
            "Structural retention decay",
            "Monetization weakness",
        ] and row["payback_health"] != "Risky":
            return "Maintain"

        if row["payback_health"] == "Risky" and row["growth_risk_level"] in ["Medium", "High"]:
            return "Reduce"

        return "Maintain"

    df["allocation_suggestion"] = df.apply(allocation_suggestion_v2, axis=1)

    # Growth health score
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

    # Measurement confidence
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

    # Final recommendation
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

    # Insights
    def generate_ai_growth_insight(row):
        if row["primary_bottleneck"] == "Weak Day-1 fit":
            return "User acquisition targeting may not match the product value proposition."
        if row["primary_bottleneck"] == "Early activation decay":
            return "Users show initial interest but fail to form early engagement habits."
        if row["primary_bottleneck"] == "Structural retention decay":
            return "User engagement weakens after the early lifecycle stage."
        if row["primary_bottleneck"] == "Activation friction":
            return "Onboarding or initial activation flow may be creating user drop-off."
        if row["primary_bottleneck"] == "Monetization weakness":
            return "Users engage with content, but conversion signals remain weak."
        if row["primary_bottleneck"] == "Capital inefficiency":
            return "Campaign cost structure may not be aligned with expected user value."
        if row["primary_bottleneck"] == "Acquisition inefficiency":
            return "Traffic quality may be too low relative to acquisition cost."
        return "Growth signals appear relatively balanced."

    def generate_ai_measurement_insight(row):
        insights = []

        if pd.isna(row["spend"]) or row["spend"] == 0:
            insights.append("Spend data is missing or zero, reducing measurement reliability.")

        if pd.isna(row["installs"]) or row["installs"] == 0:
            insights.append("Install data is missing or zero, limiting performance evaluation.")

        if str(row["os"]).strip().lower() == "ios":
            insights.append("iOS traffic may be affected by ATT-related attribution loss.")

        if row["skan_only"]:
            insights.append("SKAN-only measurement limits deterministic attribution accuracy.")

        if not insights:
            return "Measurement coverage appears relatively stable."

        return " ".join(insights)

    df["ai_growth_insight"] = df.apply(generate_ai_growth_insight, axis=1)
    df["ai_measurement_insight"] = df.apply(generate_ai_measurement_insight, axis=1)

    def build_audit_summary_v4(row):
        parts = []

        if row["traffic_efficiency"] == "Strong":
            parts.append("low CPI")
        elif row["traffic_efficiency"] == "Weak":
            parts.append("high CPI")

        if row["activation_efficiency"] == "Strong":
            parts.append("strong activation")
        elif row["activation_efficiency"] == "Weak":
            parts.append("weak activation")

        if row["early_signal_health"] == "Strong":
            parts.append("strong early signal")
        elif row["early_signal_health"] == "Weak":
            parts.append("weak early signal")

        if row["retention_stability"] == "Strong":
            parts.append("stable D7 retention")
        elif row["retention_stability"] == "Weak":
            parts.append("weak D7 retention")

        if row["revenue_efficiency"] == "Strong":
            parts.append("strong monetization")
        elif row["revenue_efficiency"] == "Weak":
            parts.append("weak monetization")

        if row["payback_health"] == "Strong":
            parts.append("healthy payback")
        elif row["payback_health"] == "Risky":
            parts.append("payback risk")

        summary_body = ", ".join(parts) if parts else "mixed performance"

        return (
            f"{summary_body}; main bottleneck: {row['primary_bottleneck']}; "
            f"growth score: {row['growth_health_score']}; "
            f"measurement confidence: {row['measurement_confidence_level']}; "
            f"final action: {row['final_recommendation_v4']}."
        )

    df["audit_summary"] = df.apply(build_audit_summary_v4, axis=1)

    # Output frames
    audit_columns = [
        "channel",
        "campaign",
        "os",
        "period_start",
        "period_end",
        "spend",
        "installs",
        "activated_users",
        "d1_retention",
        "d3_retention",
        "d7_retention",
        "revenue",
        "skan_only",
        "strategic_channel",
        "cpi",
        "activation_rate",
        "arpu",
        "early_signal_score",
        "retention_curve_quality",
        "estimated_ltv",
        "predicted_ltv",
        "payback_period",
        "predicted_payback",
        "traffic_efficiency",
        "activation_efficiency",
        "early_signal_health",
        "curve_quality_health",
        "retention_stability",
        "revenue_efficiency",
        "payback_health",
        "growth_health_score",
        "growth_health_category",
        "measurement_confidence_score",
        "measurement_confidence_level",
        "measurement_flag",
        "os_tracking_risk",
        "primary_bottleneck",
        "growth_risk_level",
        "allocation_suggestion",
        "final_recommendation_v4",
        "audit_summary",
        "ai_growth_insight",
        "ai_measurement_insight",
    ]
    audit_df = df[audit_columns].copy()

    summary = {
        "total_spend": df["spend"].sum(skipna=True),
        "total_installs": df["installs"].sum(skipna=True),
        "avg_d1_retention": df["d1_retention"].mean(skipna=True),
        "avg_d3_retention": df["d3_retention"].mean(skipna=True),
        "avg_d7_retention": df["d7_retention"].mean(skipna=True),
        "avg_early_signal_score": df["early_signal_score"].mean(skipna=True),
        "avg_cpi": df["cpi"].mean(skipna=True),
        "avg_arpu": df["arpu"].mean(skipna=True),
        "avg_payback": df["payback_period"].mean(skipna=True),
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
        avg_predicted_ltv=("predicted_ltv", "mean"),
        avg_payback=("payback_period", "mean"),
    ).reset_index()

    channel_summary["channel_cpi"] = channel_summary.apply(
        lambda x: safe_divide(x["spend"], x["installs"]), axis=1
    )

    channel_summary["avg_growth_health_score"] = channel_summary["avg_growth_health_score"].round(1)
    channel_summary["avg_measurement_confidence_score"] = channel_summary["avg_measurement_confidence_score"].round(1)
    channel_summary["avg_predicted_ltv"] = channel_summary["avg_predicted_ltv"].round(2)
    channel_summary["avg_payback"] = channel_summary["avg_payback"].round(2)
    channel_summary["channel_cpi"] = channel_summary["channel_cpi"].round(2)

    bottleneck_counts = df["primary_bottleneck"].value_counts().reset_index()
    bottleneck_counts.columns = ["primary_bottleneck", "count"]

    final_reco_counts = df["final_recommendation_v4"].value_counts().reset_index()
    final_reco_counts.columns = ["final_recommendation_v4", "count"]

    return audit_df, summary_df, channel_summary, bottleneck_counts, final_reco_counts


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.header("Upload CSV")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

st.sidebar.markdown("### Required columns")
st.sidebar.code(
    """channel
campaign
os
spend
installs
activated_users
d1_retention
d3_retention
d7_retention
revenue
skan_only
strategic_channel
period_start
period_end""",
    language="text"
)

# --------------------------------------------------
# Main
# --------------------------------------------------
if uploaded_file is None:
    st.info("CSV 파일을 업로드하면 분석이 시작됩니다.")
else:
    try:
        raw_df = pd.read_csv(uploaded_file)

        st.subheader("Raw Input Preview")
        st.dataframe(raw_df.head(), use_container_width=True)

        audit_df, summary_df, channel_summary, bottleneck_counts, final_reco_counts = run_growth_audit_v4(raw_df.copy())

        st.subheader("Executive Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Avg Growth Health Score", f"{summary_df.loc[0, 'avg_growth_health_score']:.1f}")
        c2.metric("Avg Measurement Confidence", f"{summary_df.loc[0, 'avg_measurement_confidence_score']:.1f}")
        c3.metric("Scale Count", int(summary_df.loc[0, "scale_count"]))
        c4.metric("High Risk Count", int(summary_df.loc[0, "high_risk_count"]))

        st.subheader("Summary Table")
        st.dataframe(summary_df, use_container_width=True)

        st.subheader("Campaign Audit Output")
        st.dataframe(audit_df, use_container_width=True)

        st.subheader("Channel Intelligence")
        st.dataframe(channel_summary, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Bottleneck Distribution")
            st.bar_chart(
                bottleneck_counts.set_index("primary_bottleneck")
            )

        with col2:
            st.subheader("Final Recommendation Distribution")
            st.bar_chart(
                final_reco_counts.set_index("final_recommendation_v4")
            )

        st.subheader("Download Results")
        st.download_button(
            "Download Audit Output CSV",
            audit_df.to_csv(index=False).encode("utf-8"),
            file_name="growth_audit_output_v4.csv",
            mime="text/csv",
        )

        st.download_button(
            "Download Channel Summary CSV",
            channel_summary.to_csv(index=False).encode("utf-8"),
            file_name="growth_channel_summary_v4.csv",
            mime="text/csv",
        )

    except Exception as e:
        st.error(f"Error: {e}")
