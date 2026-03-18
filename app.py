import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ページ設定
st.set_page_config(
    page_title="AI 成長監査システム (Growth Audit System) v0.4",
    layout="wide"
)

# --------------------------------------------------
# カスタムスタイル (ダークモード・モダンUI)
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
.mono-box {
    background-color: #111827;
    color: #f9fafb;
    padding: 1.2rem 1.5rem;
    border-radius: 16px;
    border: 1px solid #1f2937;
    font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
    font-size: 0.95rem;
    line-height: 1.8;
}
.playbook-title {
    color: #60a5fa;
    font-size: 1.15rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}
.action-item {
    margin-bottom: 8px;
    border-left: 3px solid #3b82f6;
    padding-left: 12px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# 必須カラム定義
# --------------------------------------------------
REQUIRED_COLUMNS = [
    "channel", "campaign", "os", "spend", "installs", "activated_users",
    "d1_retention", "d3_retention", "d7_retention", "revenue",
    "skan_only", "strategic_channel", "period_start", "period_end",
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

def map_score(value):
    score_map = {"良好": 100, "普通": 60, "注意": 30, "リスクあり": 20, "不明": 50}
    return score_map.get(value, 50)

def score_category(score):
    if pd.isna(score): return "不明"
    if score >= 80: return "健全な成長"
    if score >= 60: return "最適化の余地あり"
    if score >= 40: return "構造的な成長課題"
    return "致命的な成長リスク"

# --------------------------------------------------
# コアエンジン (分析ロジック)
# --------------------------------------------------
def run_growth_audit_v4(df: pd.DataFrame):
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"必須カラムが不足しています: {missing_cols}")

    numeric_cols = ["spend", "installs", "activated_users", "d1_retention", "d3_retention", "d7_retention", "revenue"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["d1_retention", "d3_retention", "d7_retention"]:
        df[col] = df[col].clip(lower=0, upper=0.9999)

    df["skan_only"] = df["skan_only"].apply(normalize_bool)
    df["strategic_channel"] = df["strategic_channel"].apply(normalize_bool)
    
    # 指標計算
    df["cpi"] = df.apply(lambda x: safe_divide(x["spend"], x["installs"]), axis=1)
    df["activation_rate"] = df.apply(lambda x: safe_divide(x["activated_users"], x["installs"]), axis=1)
    df["arpu"] = df.apply(lambda x: safe_divide(x["revenue"], x["installs"]), axis=1)
    df["payback_period"] = df.apply(lambda x: safe_divide(x["cpi"], x["arpu"]), axis=1)
    df["early_signal_score"] = (df["d1_retention"] * 0.2 + df["d3_retention"] * 0.3 + df["d7_retention"] * 0.5) * 100

    avg_cpi = df["cpi"].mean()
    avg_arpu = df["arpu"].mean()

    # スコアリング
    df["traffic_efficiency"] = df["cpi"].apply(lambda x: rate_relative_low_is_good(x, avg_cpi))
    df["activation_efficiency"] = df["activation_rate"].apply(lambda x: "良好" if x >= 0.5 else ("普通" if x >= 0.3 else "注意"))
    df["retention_stability"] = df["d7_retention"].apply(lambda x: "良好" if x >= 0.25 else ("普通" if x >= 0.15 else "注意"))
    df["revenue_efficiency"] = df["arpu"].apply(lambda x: rate_relative_high_is_good(x, avg_arpu))
    df["payback_health"] = df["payback_period"].apply(lambda x: "良好" if x <= 1.5 else ("普通" if x <= 3.0 else "リスクあり"))

    # ボトルネック検知
    def detect_bottleneck(row):
        if row["d1_retention"] < 0.20: return "Day-1 適合性の不足"
        if (row["d1_retention"] - row["d3_retention"]) > 0.20: return "初期アクティベーションの離脱"
        if (row["d3_retention"] - row["d7_retention"]) > 0.15: return "構造的な継続率の低下"
        if row["payback_health"] == "リスクあり": return "資本効率の悪化"
        return "重大なボトルネックなし"

    df["primary_bottleneck"] = df.apply(detect_bottleneck, axis=1)
    
    # スコア算出
    df["growth_health_score"] = (
        df["traffic_efficiency"].map(map_score) * 0.10 +
        df["activation_efficiency"].map(map_score) * 0.15 +
        df["retention_stability"].map(map_score) * 0.40 +
        df["revenue_efficiency"].map(map_score) * 0.20 +
        df["payback_health"].map(map_score) * 0.15
    ).round(1)

    # 計測信頼性
    def get_conf(row):
        score = 100
        if row["spend"] == 0: score -= 50
        if str(row["os"]).lower() == "ios": score -= 15
        return max(score, 0)
    df["measurement_confidence_score"] = df.apply(get_conf, axis=1)

    # 推奨アクション
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
# メ인 UI 렌더링
# --------------------------------------------------
st.markdown("## AI 成長監査システム (Growth Audit System)")
st.sidebar.markdown("### データアップロード")
uploaded_file = st.sidebar.file_uploader("CSVファイルをアップロード", type=["csv"])

if uploaded_file:
    audit_df, summary_df = run_growth_audit_v4(pd.read_csv(uploaded_file))

    # フィルター
    st.sidebar.markdown("### フィルター設定")
    selected_ch = st.sidebar.selectbox("チャネル選択", ["すべて"] + sorted(audit_df["channel"].unique().tolist()))
    
    filtered_df = audit_df if selected_ch == "すべて" else audit_df[audit_df["channel"] == selected_ch]

    # KPI表示
    st.markdown("### エグゼクティブ・オーバービュー")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("平均成長スコア", f"{summary_df.loc[0, 'avg_growth_health_score']:.1f}")
    k2.metric("計測信頼性", f"{summary_df.loc[0, 'avg_measurement_confidence_score']:.1f}")
    k3.metric("高リスク件数", int(summary_df.loc[0, "high_risk_count"]))
    k4.metric("拡大推奨数", int(summary_df.loc[0, "scale_count"]))

    # --------------------------------------------------
    # 高度な AI Playbook ロジック
    # --------------------------------------------------
    st.markdown("### 🤖 AI 戦略・運用プレイブック")
    
    if not filtered_df.empty:
        main_bn = filtered_df["primary_bottleneck"].mode()[0]
        avg_cpi = filtered_df["cpi"].mean()
        avg_d1 = filtered_df["d1_retention"].mean()
        
        playbook_config = {
            "Day-1 適合性の不足": {
                "title": "流入クオリティの最適化 (Creative-Product Fit)",
                "analysis": f"現在のD1継続率は {avg_d1:.1%} です。これは広告の訴求内容とアプリ体験に乖離があることを示唆しています。",
                "actions": [
                    "<b>クリエイティブの整合性:</b> 広告で使用している作品を、ホーム画面の最上部バナーに固定配置してください。",
                    "<b>ターゲットの精査:</b> ジャンル特化型から、汎用性の高い人気作品（ヒット作）中心の運用へ切り替えてください。",
                    "<b>OS別戦略:</b> iOSの継続率が低い場合、ATTポップアップの表示タイミングを『作品読了後』に遅らせる調整を推奨します。"
                ]
            },
            "初期アクティベーションの離脱": {
                "title": "オンボーディングの摩擦除去 (Onboarding Optimization)",
                "analysis": "インストール直後のユーザーが、コンテンツを楽しむ前に離脱しています。UX上の障壁が存在します。",
                "actions": [
                    "<b>ディープリンク改善:</b> 作品詳細ページを経由せず、直接『第1話の閲覧画面』へ遷移する運用をテストしてください。",
                    "<b>即時報酬の提供:</b> 1話読了時に、即座に利用可能な『無料チケット』をPush通知で自動配布してください。",
                    "<b>導線簡素化:</b> 起動時のポップアップ露出を30%削減し、読書開始までのクリック数を最小化してください。"
                ]
            },
            "資本効率の悪化": {
                "title": "予算配分と収益性の改善 (Budget Rebalancing)",
                "analysis": f"現在の平均CPIは ¥{avg_cpi:.0f} です。獲得コストが将来収益（LTV）を圧迫しています。",
                "actions": [
                    "<b>高コストキャンペーンの削減:</b> 平均CPIを25%以上上回るキャンペーンの予算を即座に30%削減してください。",
                    "<b>課金トリガーの露出:</b> コインチャージ画面への遷移率を確認し、初回限定の『ウェルカムパック』を強調表示してください。",
                    "<b>イベント最適化:</b> インストール最大化から、『課金完了（Purchase）』を最適化基準とした配信への移行を検討してください。"
                ]
            },
            "構造的な継続率の低下": {
                "title": "長期定着（LTV）の構造的改善",
                "analysis": "数日間の利用後、ユーザーが習慣化する前にアプリを離れています。連載作品への誘導に課題があります。",
                "actions": [
                    "<b>長期連載への誘導:</b> 短編よりも、100話以上の話数を持つ『長期連載作品』への誘導比率を高めてください。",
                    "<b>リテンション通知:</b> お気に入り登録済み作品の更新通知の到達率を再点検し、開封率の低い時間は配信を避けてください。",
                    "<b>再訪問ボーナス:</b> 7日間未ログインのユーザーに対し、再ログイン時限定のボーナス付与キャンペーンを自動化してください。"
                ]
            }
        }

        guide = playbook_config.get(main_bn, {
            "title": "健全な成長 (Keep Scaling)",
            "analysis": "顕著なボトルネックは見当たりません。現在の戦略を維持しつつ、予算を拡大するフェーズです。",
            "actions": [
                "<b>スケーリング:</b> 高効率なキャンペーンの予算を週次で15%ずつ増額してください。",
                "<b>クリエイティブ展開:</b> 現在成功しているクリエイティブの要素を、他の作品にも横展開してください。"
            ]
        })

        st.markdown(f"""
        <div class="mono-box">
            <div class="playbook-title">🎯 重点改善タスク: {guide['title']}</div>
            <div style="color: #9ca3af; margin-bottom: 15px;">📊 分析結果: {guide['analysis']}</div>
            <div style="font-weight: 600; margin-bottom: 8px;">🚀 推奨アクション (Action Items):</div>
            {"".join([f'<div class="action-item">{a}</div>' for a in guide['actions']])}
            <div style="margin-top: 15px; font-size: 0.85rem; color: #f87171;">
                ※ 本提案は開発リソースを必要とせず、<b>運用設定のみで即座に実行可能</b>です。
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 監査テーブル
    st.markdown("### キャンペーン監査テーブル")
    styled_df = filtered_df[["channel", "campaign", "os", "growth_health_score", "primary_bottleneck", "final_recommendation_v4"]].style.applymap(highlight_growth_score, subset=["growth_health_score"])
    st.dataframe(styled_df, use_container_width=True)

    # 散布図
    st.markdown("### 戦略的散布図")
    scatter = alt.Chart(filtered_df).mark_circle(size=100).encode(
        x=alt.X('growth_health_score', title='成長ヘルス'),
        y=alt.X('measurement_confidence_score', title='計測信頼性'),
        color='final_recommendation_v4',
        tooltip=['campaign', 'growth_health_score', 'primary_bottleneck']
    ).properties(height=400).interactive()
    st.altair_chart(scatter, use_container_width=True)

else:
    st.info("左側のサイドバーからCSVファイルをアップロードしてください。")
