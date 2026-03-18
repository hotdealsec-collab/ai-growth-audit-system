import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# 1. ページ設定
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

# スタイル関数
def highlight_growth_score(val):
    if pd.isna(val): return ""
    if val < 50: return "background-color: rgba(239, 68, 68, 0.35); color: white;"
    if val < 60: return "background-color: rgba(245, 158, 11, 0.30); color: white;"
    return ""

def map_score(value):
    score_map = {"良好": 100, "普通": 60, "注意": 30, "リスクあり": 20, "不明": 50}
    return score_map.get(value, 50)

# --------------------------------------------------
# 4. コアエンジン (分析ロジック)
# --------------------------------------------------
def run_growth_audit_v4(df: pd.DataFrame):
    # データクリーニング
    numeric_cols = ["spend", "installs", "activated_users", "d1_retention", "d3_retention", "d7_retention", "revenue"]
    for col in numeric_cols: df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["d1_retention", "d3_retention", "d7_retention"]: df[col] = df[col].clip(lower=0, upper=0.9999)

    df["skan_only"] = df["skan_only"].apply(normalize_bool)
    
    # 指標算出
    df["cpi"] = df.apply(lambda x: safe_divide(x["spend"], x["installs"]), axis=1)
    df["arpu"] = df.apply(lambda x: safe_divide(x["revenue"], x["installs"]), axis=1)
    df["payback_period"] = df.apply(lambda x: safe_divide(x["cpi"], x["arpu"]), axis=1)

    avg_cpi = df["cpi"].mean()
    avg_arpu = df["arpu"].mean()

    # 効率性判定
    df["traffic_efficiency"] = df["cpi"].apply(lambda x: "良好" if x <= avg_cpi * 0.85 else ("普通" if x <= avg_cpi * 1.15 else "注意"))
    df["activation_rate"] = df["activated_users"] / df["installs"]
    df["activation_efficiency"] = df["activation_rate"].apply(lambda x: "良好" if x >= 0.5 else ("普通" if x >= 0.3 else "注意"))
    df["retention_stability"] = df["d7_retention"].apply(lambda x: "良好" if x >= 0.25 else ("普通" if x >= 0.15 else "注意"))
    df["revenue_efficiency"] = df["arpu"].apply(lambda x: "良好" if x >= avg_arpu * 1.15 else ("普通" if x >= avg_arpu * 0.85 else "注意"))
    df["payback_health"] = df["payback_period"].apply(lambda x: "良好" if x <= 1.5 else ("普通" if x <= 3.0 else "リスクあり"))

    # ボトルネック判定
    def detect_bottleneck(row):
        if row["d1_retention"] < 0.20: return "Day-1 適合性の不足"
        if (row["d1_retention"] - row["d3_retention"]) > 0.20: return "初期アクティベーションの離脱"
        if (row["d3_retention"] - row["d7_retention"]) > 0.15: return "構造的な継続率の低下"
        if row["payback_health"] == "リスクあり": return "資本効率の悪化"
        return "重大なボトルネックなし"

    df["primary_bottleneck"] = df.apply(detect_bottleneck, axis=1)
    
    # スコア計算
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

    return df

# --------------------------------------------------
# 5. UI レンダリング
# --------------------------------------------------
st.sidebar.markdown("### 📥 アップロード")
uploaded_file = st.sidebar.file_uploader("CSVファイルをアップロード", type=["csv"])

if uploaded_file:
    try:
        raw_df = pd.read_csv(uploaded_file)
        audit_df = run_growth_audit_v4(raw_df)

        # --- サイドバー フィルター ---
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 🔍 フィルター")
        channels = ["すべて"] + sorted(audit_df["channel"].unique().tolist())
        selected_ch = st.sidebar.selectbox("チャネル選択", channels)
        
        filtered_df = audit_df if selected_ch == "すべて" else audit_df[audit_df["channel"] == selected_ch]

        # --- エグゼクティブ・オーバービュー ---
        st.markdown("### 📊 エグゼクティブ・オーバービュー")
        k1, k2, k3, k4 = st.columns(4)
        
        avg_score = filtered_df["growth_health_score"].mean()
        avg_conf = filtered_df["measurement_confidence_score"].mean()
        high_risk = (filtered_df["payback_health"] == "リスクあり").sum()
        scale_count = (filtered_df["final_recommendation_v4"] == "拡大 (Scale)").sum()

        k1.metric("平均成長スコア", f"{avg_score:.1f}")
        k2.metric("計測信頼性", f"{avg_conf:.1f}")
        k3.metric("高リスク件数", int(high_risk))
        k4.metric("拡大推奨数", int(scale_count))

        # --- AI 戦略プレイブック ---
        st.markdown("### 🤖 AI 戦略・運用プレイブック")
        if not filtered_df.empty:
            main_bn = filtered_df["primary_bottleneck"].mode()[0]
            avg_cpi = filtered_df["cpi"].mean()
            avg_d1 = filtered_df["d1_retention"].mean()
            
            playbook_config = {
                "Day-1 適合性の不足": {
                    "title": "流入クオリティの最適化 (Creative-Product Fit)",
                    "analysis": f"現在のD1継続率は {avg_d1:.1%} です。ターゲット層とコンテンツが合致していません。",
                    "actions": [
                        "<b>クリエイティブの整合性:</b> 広告クリエイティブの作品を、アプリ内ホーム画面の最上部に固定表示してください。",
                        "<b>ターゲットの再定義:</b> ジャンル特化ではなく、汎用性の高い人気作品（ヒット作）中心の運用へ移行してください。",
                        "<b>iOS対応:</b> ATT同意ポップアップの表示タイミングを『作品の1話読了後』に変更することを推奨します。"
                    ]
                },
                "初期アクティベーションの離脱": {
                    "title": "オンボーディングの摩擦除去 (UX Optimization)",
                    "analysis": "新規ユーザーが本格的な読書体験（Aha-moment）に到達する前に離脱しています。",
                    "actions": [
                        "<b>ディープリンクの活用:</b> 作品詳細ページをスキップし、直接『第1話の閲覧画面』へ誘導する設定をテストしてください。",
                        "<b>即時報酬の提供:</b> 1話読了の瞬間に、追加で利用可能なチケットをPush通知で配布してください。",
                        "<b>UIの簡素化:</b> 起動直後のポップアップ露出を削減し、コンテンツまでのクリック数を最小化してください。"
                    ]
                },
                "資本効率の悪化": {
                    "title": "予算配分と収益性の改善 (Budget Rebalancing)",
                    "analysis": f"平均CPIが ¥{avg_cpi:.0f} と高く、将来的な収益（LTV）が獲得コストを下回るリスクがあります。",
                    "actions": [
                        "<b>予算の再配分:</b> 平均CPIを20%以上上回るキャンペーンの予算を削減し、高効率なメディアへ再配分してください。",
                        "<b>課金トリガーの露出:</b> ショップ画面のUIを確認し、初回限定の『コイン増量パッケージ』を強調してください。",
                        "<b>イベント最適化:</b> インストール最大化ではなく、『課金完了（Purchase）』を最適化指標に変更してください。"
                    ]
                },
                "構造的な継続率の低下": {
                    "title": "長期定着（LTV）の構造的改善",
                    "analysis": "数日間の利用後に離脱が目立ちます。連載作品への定着や習慣化に課題があります。",
                    "actions": [
                        "<b>長期連載への誘導:</b> 短編よりも、話数が多い『長期連載作品』への誘導比率を高めてください。",
                        "<b>通知設定の最適化:</b> お気に入り登録済み作品の更新通知の開封率を点検し、送信タイミングをパーソナライズしてください。",
                        "<b>休眠防止策:</b> 未ログイン期間が3日を超えたユーザーに対し、再ログイン限定ボーナスを自動送付してください。"
                    ]
                }
            }
            guide = playbook_config.get(main_bn, {
                "title": "健全な成長 (Keep Scaling)",
                "analysis": "現在、大きなボトルネックは見当たりません。スケーリングのチャンスです。",
                "actions": ["<b>スケーリング:</b> 成功しているキャンペーンの予算を週次で15%ずつ増額し、リーチを拡大してください。"]
            })

            st.markdown(f"""
            <div class="mono-box">
                <div class="playbook-title">🎯 重点改善タスク: {guide['title']}</div>
                <div style="color: #9ca3af; margin-bottom: 15px;">📊 分析結果: {guide['analysis']}</div>
                {"".join([f'<div class="action-item">{a}</div>' for a in guide['actions']])}
            </div>
            """, unsafe_allow_html=True)

        # --- キャンペーン監査テーブル ---
        st.markdown("### 📋 キャンペーン監査テーブル")
        view_cols = ["channel", "campaign", "os", "growth_health_score", "primary_bottleneck", "final_recommendation_v4"]
        
        # map関数を使用してスタイルを適用
        styled_df = filtered_df[view_cols].style.map(highlight_growth_score, subset=["growth_health_score"])
        st.dataframe(styled_df, use_container_width=True)

        # --- チャート ---
        st.markdown("### 📈 戦略的散布図")
        scatter = alt.Chart(filtered_df).mark_circle(size=120).encode(
            x=alt.X('growth_health_score', title='成長ヘルススコア'),
            y=alt.Y('measurement_confidence_score', title='計測信頼性スコア'),
            color=alt.Color('final_recommendation_v4', title='推奨アクション'),
            tooltip=['campaign', 'growth_health_score', 'measurement_confidence_score', 'primary_bottleneck']
        ).properties(height=450).interactive()
        st.altair_chart(scatter, use_container_width=True)

    except Exception as e:
        st.error(f"分析中にエラーが発生しました: {e}")
else:
    st.info("👈 左側のサイドバーからCSVファイルをアップロードしてください。")
