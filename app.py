import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# 1. 페이지 설정
st.set_page_config(
    page_title="AI 成長監査システム (Growth Audit System) v0.4",
    layout="wide"
)

# --------------------------------------------------
# 2. カスタムスタイル (CSS)
# --------------------------------------------------
st.markdown("""
<style>
.block-container { padding-top: 2rem; padding-bottom: 2rem; }
.small-note { color: #6b7280; font-size: 0.9rem; }
.mono-box {
    background-color: #111827; color: #f9fafb; padding: 1.2rem 1.5rem;
    border-radius: 16px; border: 1px solid #1f2937;
    font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
    font-size: 0.95rem; line-height: 1.8;
}
.playbook-title { color: #60a5fa; font-size: 1.15rem; font-weight: 700; margin-bottom: 0.5rem; }
.action-item { margin-bottom: 8px; border-left: 3px solid #3b82f6; padding-left: 12px; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# 3. 必須カラム及びヘルパー関数
# --------------------------------------------------
REQUIRED_COLUMNS = [
    "channel", "campaign", "os", "spend", "installs", "activated_users",
    "d1_retention", "d3_retention", "d7_retention", "revenue",
    "skan_only", "strategic_channel", "period_start", "period_end",
]

def safe_divide(a, b):
    if pd.isna(a) or pd.isna(b) or b == 0: return np.nan
    return a / b

def normalize_bool(value):
    if pd.isna(value): return False
    return str(value).strip().lower() in ["true", "1", "yes", "y"]

# 스타일링 함수 (에러 해결 핵심)
def highlight_growth_score(val):
    if pd.isna(val): return ""
    if val < 50: return "background-color: rgba(239, 68, 68, 0.35); color: white;"
    if val < 60: return "background-color: rgba(245, 158, 11, 0.30); color: white;"
    return ""

def map_score(value):
    score_map = {"良好": 100, "普通": 60, "注意": 30, "リスクあり": 20, "不明": 50}
    return score_map.get(value, 50)

def score_category(score):
    if pd.isna(score): return "不明"
    if score >= 80: return "健全な成長"
    if score >= 60: return "最適化の余지あり"
    if score >= 40: return "構造的な成長課題"
    return "致命的な成長リスク"

# --------------------------------------------------
# 4. コアエンジン (分析ロジック)
# --------------------------------------------------
def run_growth_audit_v4(df: pd.DataFrame):
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols: raise ValueError(f"必須カラム不足: {missing_cols}")

    numeric_cols = ["spend", "installs", "activated_users", "d1_retention", "d3_retention", "d7_retention", "revenue"]
    for col in numeric_cols: df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["d1_retention", "d3_retention", "d7_retention"]: df[col] = df[col].clip(lower=0, upper=0.9999)

    df["skan_only"] = df["skan_only"].apply(normalize_bool)
    df["cpi"] = df.apply(lambda x: safe_divide(x["spend"], x["installs"]), axis=1)
    df["arpu"] = df.apply(lambda x: safe_divide(x["revenue"], x["installs"]), axis=1)
    df["payback_period"] = df.apply(lambda x: safe_divide(x["cpi"], x["arpu"]), axis=1)

    avg_cpi = df["cpi"].mean()
    avg_arpu = df["arpu"].mean()

    # 효율성 판정 로직
    df["traffic_efficiency"] = df["cpi"].apply(lambda x: "良好" if x <= avg_cpi * 0.85 else ("普通" if x <= avg_cpi * 1.15 else "注意"))
    df["activation_efficiency"] = df["activated_users"] / df["installs"]
    df["activation_efficiency"] = df["activation_efficiency"].apply(lambda x: "良好" if x >= 0.5 else ("普通" if x >= 0.3 else "注意"))
    df["retention_stability"] = df["d7_retention"].apply(lambda x: "良好" if x >= 0.25 else ("普通" if x >= 0.15 else "注意"))
    df["revenue_efficiency"] = df["arpu"].apply(lambda x: "良好" if x >= avg_arpu * 1.15 else ("普通" if x >= avg_arpu * 0.85 else "注意"))
    df["payback_health"] = df["payback_period"].apply(lambda x: "良好" if x <= 1.5 else ("普通" if x <= 3.0 else "リスクあり"))

    # 보틀넥 판정
    def detect_bottleneck(row):
        if row["d1_retention"] < 0.20: return "Day-1 適合性の不足"
        if (row["d1_retention"] - row["d3_retention"]) > 0.20: return "初期アクティベーションの離脱"
        if (row["d3_retention"] - row["d7_retention"]) > 0.15: return "構造적인継続率の低下"
        if row["payback_health"] == "リスクあり": return "資本効率の悪化"
        return "重大なボトルネックなし"

    df["primary_bottleneck"] = df.apply(detect_bottleneck, axis=1)
    
    # 종합 스코어
    df["growth_health_score"] = (
        df["traffic_efficiency"].map(map_score) * 0.10 +
        df["activation_efficiency"].map(map_score) * 0.15 +
        df["retention_stability"].map(map_score) * 0.40 +
        df["revenue_efficiency"].map(map_score) * 0.20 +
        df["payback_health"].map(map_score) * 0.15
    ).round(1)

    def get_conf(row):
        score = 100
        if row["spend"] == 0: score -= 50
        if str(row["os"]).lower() == "ios": score -= 15
        return max(score, 0)
    df["measurement_confidence_score"] = df.apply(get_conf, axis=1)

    def get_reco(row):
        if row["growth_health_score"] >= 75: return "拡大 (Scale)"
        if row["growth_health_score"] < 50: return "縮小 (Reduce)"
        return "維持 (Maintain)"
    df["final_recommendation_v4"] = df.apply(get_reco, axis=1)

    summary_df = pd.DataFrame([{
        "avg_growth_health_score": df["growth_health_score"].mean(),
        "avg_measurement_confidence_score": df["measurement_confidence_score"].mean(),
        "high_risk_count": (df["payback_health"] == "リスクあり").sum(),
        "scale_count": (df["final_recommendation_v4"] == "拡大 (Scale)").sum()
    }])

    return df, summary_df

# --------------------------------------------------
# 5. UI 및 대시보드
# --------------------------------------------------
st.markdown("## AI 成長監査システム (Growth Audit System)")
st.sidebar.markdown("### アップロード")
uploaded_file = st.sidebar.file_uploader("CSVをアップロード", type=["csv"])

if uploaded_file:
    try:
        raw_data = pd.read_csv(uploaded_file)
        audit_df, summary_df = run_growth_audit_v4(raw_data)

        # 사이드바 필터
        st.sidebar.markdown("### フィルター")
        channels = ["すべて"] + sorted(audit_df["channel"].unique().tolist())
        selected_ch = st.sidebar.selectbox("チャネル選択", channels)
        filtered_df = audit_df if selected_ch == "すべて" else audit_df[audit_df["channel"] == selected_ch]

        # KPI Metrics
        st.markdown("### エグゼクティブ・オーバービュー")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("平均成長スコア", f"{summary_df.loc[0, 'avg_growth_health_score']:.1f}")
        k2.metric("計測信頼性", f"{summary_df.loc[0, 'avg_measurement_confidence_score']:.1f}")
        k3.metric("高リスク件数", int(summary_df.loc[0, "high_risk_count"]))
        k4.metric("拡大推奨数", int(summary_df.loc[0, "scale_count"]))

        # AI Playbook
        st.markdown("### 🤖 AI 戦略・運用プレイブック")
        if not filtered_df.empty:
            main_bn = filtered_df["primary_bottleneck"].mode()[0]
            avg_cpi = filtered_df["cpi"].mean()
            avg_d1 = filtered_df["d1_retention"].mean()
            
            playbook_config = {
                "Day-1 適合性の不足": {
                    "title": "流入クオリティの最適化 (Creative-Product Fit)",
                    "analysis": f"現在のD1継続率は {avg_d1:.1%} です。広告とアプリ体験に乖離があります。",
                    "actions": [
                        "<b>クリエイティブの整合性:</b> 広告作品をホーム画面最上部に固定配置してください。",
                        "<b>ターゲットの精査:</b> 汎用性の高い人気作品中心の運用へ切り替えてください。",
                        "<b>iOS対応:</b> ATTポップアップ表示を『読了後』に遅らせる調整を推奨します。"
                    ]
                },
                "初期アクティベーションの離脱": {
                    "title": "オンボーディングの摩擦除去 (UX Optimization)",
                    "analysis": "インストール直後のユーザーが、コンテンツを楽しむ前に離脱しています。",
                    "actions": [
                        "<b>ディープリンク改善:</b> 直接『第1話閲覧画面』へ遷移する運用をテストしてください。",
                        "<b>即時報酬:</b> 1話読了時にチケットをPush通知で自動配布してください。",
                        "<b>導線簡素化:</b> 起動時のポップアップを削減し、読書までの手順を最小化してください。"
                    ]
                },
                "資本効率の悪化": {
                    "title": "予算配分と収益性の改善 (Budget Rebalancing)",
                    "analysis": f"平均CPIは ¥{avg_cpi:.0f} です。獲得コストが収益を圧迫しています。",
                    "actions": [
                        "<b>予算削減:</b> CPIが高いキャンペーンの予算を即座に30%削減してください。",
                        "<b>課金誘導:</b> 初回限定『ウェルカムパック』をショップで強調表示してください。",
                        "<b>最適化変更:</b> 『課金完了(Purchase)』を最適化基準とした配信を検討してください。"
                    ]
                },
                "構造적인継続率の低下": {
                    "title": "長期定着（LTV）の構造的改善",
                    "analysis": "数日利用後、習慣化する前に離れています。連載誘導に課題があります。",
                    "actions": [
                        "<b>長期連載への誘導:</b> 100話以上の話数を持つ『長期連載作品』への誘導比率を高めてください。",
                        "<b>通知再点検:</b> お気に入り作品の更新通知の到達率と開封時間を最適化してください。",
                        "<b>再訪問ボーナス:</b> 7日間未ログインユーザーへのボーナス自動付与をガ동하십시오."
                    ]
                }
            }
            guide = playbook_config.get(main_bn, {
                "title": "健全な成長 (Keep Scaling)",
                "analysis": "顕著なボトルネックはありません。現在の戦略を維持しつつ拡大してください。",
                "actions": ["<b>スケーリング:</b> 高効率キャンペーン予算を週次15%ずつ増額。"]
            })

            st.markdown(f"""
            <div class="mono-box">
                <div class="playbook-title">🎯 重点改善タスク: {guide['title']}</div>
                <div style="color: #9ca3af; margin-bottom: 15px;">📊 分析結果: {guide['analysis']}</div>
                {"".join([f'<div class="action-item">{a}</div>' for a in guide['actions']])}
            </div>
            """, unsafe_allow_html=True)

        # 監査テーブル (Error 해결된 부분)
        st.markdown("### キャンペーン監査テーブル")
        view_cols = ["channel", "campaign", "os", "growth_health_score", "primary_bottleneck", "final_recommendation_v4"]
        
        # Pandas 최신 버전 호환을 위해 map 사용 (highlight_growth_score 함수 호출)
        styled_df = filtered_df[view_cols].style.map(highlight_growth_score, subset=["growth_health_score"])
        st.dataframe(styled_df, use_container_width=True)

        # Charts
        st.markdown("### 戦略的散布図")
        scatter = alt.Chart(filtered_df).mark_circle(size=100).encode(
            x=alt.X('growth_health_score', title='成長ヘルス'),
            y=alt.Y('measurement_confidence_score', title='計測信頼性'),
            color='final_recommendation_v4',
            tooltip=['campaign', 'growth_health_score', 'primary_bottleneck']
        ).properties(height=400).interactive()
        st.altair_chart(scatter, use_container_width=True)

    except Exception as e:
        st.error(f"分析中にエラーが発生しました: {e}")
else:
    st.info("サイドバーからCSVファイルをアップロードしてください。")
