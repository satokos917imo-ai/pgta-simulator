import streamlit as st

# ==========================================
# 🔒 パスワードロック機能（一番上に配置する）
# ==========================================
# まだ認証されていない場合は初期状態にセット
if "password_correct" not in st.session_state:
    st.session_state["password_correct"] = False

# パスワードが通っていない時の画面表示
if not st.session_state["password_correct"]:
    st.title("PGT-A費用対効果シミュレーター")
    st.info("アクセスするにはパスワードを入力してください。")
    
    password = st.text_input("パスワード", type="password")
    
    if st.button("ログイン"):
        # ⚠️ 以下の "pgta2026" の部分をお好きなパスワードに変更してください
        if password == "pgta2026": 
            st.session_state["password_correct"] = True
            st.rerun() # 認証成功後、画面をリロードして本編を表示
        else:
            st.error("😕 パスワードが違います")
    
    # 認証されるまで、これより下のプログラムは一切読み込まれません（完全に隠蔽されます）
    st.stop()
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import math

# --- 1. 年齢別データベース（25歳〜45歳） ---
# 染色体正常胚率(euploid_rate)はFranasiak et al. (2014)等の大標本データに基づき設定
db_age_data = {
    25: {"euploid_rate": 0.70, "lbr_no_pgta": 0.50, "misc_no_pgta": 0.12, "lbr_pgta": 0.68, "misc_pgta": 0.060},
    26: {"euploid_rate": 0.69, "lbr_no_pgta": 0.49, "misc_no_pgta": 0.12, "lbr_pgta": 0.68, "misc_pgta": 0.060},
    27: {"euploid_rate": 0.68, "lbr_no_pgta": 0.48, "misc_no_pgta": 0.13, "lbr_pgta": 0.67, "misc_pgta": 0.065},
    28: {"euploid_rate": 0.67, "lbr_no_pgta": 0.47, "misc_no_pgta": 0.14, "lbr_pgta": 0.67, "misc_pgta": 0.065},
    29: {"euploid_rate": 0.66, "lbr_no_pgta": 0.46, "misc_no_pgta": 0.14, "lbr_pgta": 0.66, "misc_pgta": 0.070},
    30: {"euploid_rate": 0.65, "lbr_no_pgta": 0.45, "misc_no_pgta": 0.15, "lbr_pgta": 0.65, "misc_pgta": 0.070},
    31: {"euploid_rate": 0.63, "lbr_no_pgta": 0.44, "misc_no_pgta": 0.16, "lbr_pgta": 0.65, "misc_pgta": 0.070},
    32: {"euploid_rate": 0.61, "lbr_no_pgta": 0.43, "misc_no_pgta": 0.17, "lbr_pgta": 0.64, "misc_pgta": 0.070},
    33: {"euploid_rate": 0.59, "lbr_no_pgta": 0.42, "misc_no_pgta": 0.18, "lbr_pgta": 0.64, "misc_pgta": 0.070},
    34: {"euploid_rate": 0.57, "lbr_no_pgta": 0.41, "misc_no_pgta": 0.19, "lbr_pgta": 0.63, "misc_pgta": 0.075},
    35: {"euploid_rate": 0.55, "lbr_no_pgta": 0.40, "misc_no_pgta": 0.20, "lbr_pgta": 0.63, "misc_pgta": 0.075},
    36: {"euploid_rate": 0.50, "lbr_no_pgta": 0.36, "misc_no_pgta": 0.22, "lbr_pgta": 0.61, "misc_pgta": 0.080},
    37: {"euploid_rate": 0.45, "lbr_no_pgta": 0.32, "misc_no_pgta": 0.25, "lbr_pgta": 0.60, "misc_pgta": 0.080},
    38: {"euploid_rate": 0.40, "lbr_no_pgta": 0.29, "misc_no_pgta": 0.30, "lbr_pgta": 0.60, "misc_pgta": 0.085},
    39: {"euploid_rate": 0.34, "lbr_no_pgta": 0.24, "misc_no_pgta": 0.35, "lbr_pgta": 0.60, "misc_pgta": 0.090},
    40: {"euploid_rate": 0.28, "lbr_no_pgta": 0.20, "misc_no_pgta": 0.40, "lbr_pgta": 0.55, "misc_pgta": 0.100},
    41: {"euploid_rate": 0.23, "lbr_no_pgta": 0.15, "misc_no_pgta": 0.48, "lbr_pgta": 0.55, "misc_pgta": 0.100},
    42: {"euploid_rate": 0.18, "lbr_no_pgta": 0.10, "misc_no_pgta": 0.55, "lbr_pgta": 0.50, "misc_pgta": 0.120},
    43: {"euploid_rate": 0.13, "lbr_no_pgta": 0.07, "misc_no_pgta": 0.60, "lbr_pgta": 0.48, "misc_pgta": 0.130},
    44: {"euploid_rate": 0.09, "lbr_no_pgta": 0.05, "misc_no_pgta": 0.65, "lbr_pgta": 0.45, "misc_pgta": 0.150},
    45: {"euploid_rate": 0.05, "lbr_no_pgta": 0.02, "misc_no_pgta": 0.70, "lbr_pgta": 0.40, "misc_pgta": 0.200},
}

# --- 2. 計算モデル・減衰関数（ベイズ更新ロジック） ---

def predict_oocytes_moon(age, amh):
    """Moon KY, et al. (2016) の回帰式による期待採卵数の予測"""
    log_oocytes = 3.21 - (0.036 * age) + (0.089 * amh)
    predicted = math.exp(log_oocytes)
    return min(30, max(1, int(predicted)))

def get_adjusted_lbr_pirtea(base_lbr, failed_transfers):
    """PGT-Aあり：Pirteaら(2020)のデータに基づくL字型減衰。正常胚移植不成功は母体因子を強く示唆"""
    if failed_transfers == 0:
        return base_lbr 
    elif failed_transfers == 1:
        return base_lbr * 0.84 
    else:
        return base_lbr * 0.83 

def get_adjusted_lbr_no_pgta(base_lbr, failed_transfers):
    """PGT-Aなし：英国HFEA等の統計に基づく階段状減衰。回数とともに着床不全の疑いを強化"""
    if failed_transfers == 0:
        return base_lbr 
    elif failed_transfers == 1:
        return base_lbr * 0.95 
    elif failed_transfers == 2:
        return base_lbr * 0.90 
    elif failed_transfers == 3:
        return base_lbr * 0.80 
    else:
        return base_lbr * 0.70 

def calc_collection_cycle_cost_100(eggs, blasts):
    """保険点数ベースの費用計算（10割負担額）"""
    if eggs == 0: return 32000
    c_base = 32000
    if eggs == 1: c_egg = 24000
    elif 2 <= eggs <= 5: c_egg = 36000
    elif 6 <= eggs <= 9: c_egg = 55000
    else: c_egg = 72000
    if eggs == 1: c_icsi = 48000
    elif 2 <= eggs <= 5: c_icsi = 68000
    elif 6 <= eggs <= 9: c_icsi = 100000
    else: c_icsi = 128000
    c_fert = (42000 + c_icsi) / 2
    if eggs == 1: c_cult = 45000
    elif 2 <= eggs <= 5: c_cult = 60000
    elif 6 <= eggs <= 9: c_cult = 84000
    else: c_cult = 105000
    if eggs == 1: c_blast = 15000
    elif 2 <= eggs <= 5: c_blast = 20000
    elif 6 <= eggs <= 9: c_blast = 25000
    else: c_blast = 30000
    if blasts == 0: c_freeze = 0
    elif blasts == 1: c_freeze = 50000
    elif 2 <= blasts <= 5: c_freeze = 70000
    elif 6 <= blasts <= 9: c_freeze = 102000
    else: c_freeze = 130000
    return c_base + c_egg + c_fert + c_cult + c_blast + c_freeze

# --- 3. メイン・シミュレーションロジック ---
def simulate_ivf(current_age, start_age, expected_eggs, fert_rate, blast_rate, pgta_mode, cost_coll_self, cost_trans_self, cost_pgta_unit, past_transfers=0, is_pcos=False, num_trials=1000):
    # 成功率等の医学的データは「現在の年齢（current_age）」を参照
    db = db_age_data[current_age]
    
    mature_rate = 0.72 if is_pcos else 0.83
    base_blast_yield = mature_rate * fert_rate * blast_rate * 0.75 
    euploid_rate = db["euploid_rate"]
    
    # 保険の回数制限は「開始時の年齢（start_age）」で判定
    if start_age < 40: insurance_limit = 6
    elif start_age <= 42: insurance_limit = 3
    else: insurance_limit = 0
    
    if pgta_mode:
        base_lbr = db["lbr_pgta"]
        misc_rate = db["misc_pgta"]
    else:
        base_lbr = db["lbr_no_pgta"]
        misc_rate = db["misc_no_pgta"]
        
    cost_transfer_insurance = 120000 * 0.3
    cost_miscarriage = 30000 
    
    results = []
    
    for _ in range(num_trials):
        total_time = 0
        total_collections = 0
        insurance_collections = 0
        self_collections = 0
        total_transfers = 0
        insurance_transfers = 0 
        self_transfers = 0
        cost_insurance = 0
        cost_self = 0
        success = False
        
        while not success:
            total_time += 2 
            total_collections += 1
            num_eggs = np.random.poisson(expected_eggs)
            blasts = np.random.binomial(num_eggs, base_blast_yield) if num_eggs > 0 else 0
            
            if pgta_mode:
                cost_self += cost_coll_self
                self_collections += 1
            else:
                if (past_transfers + insurance_transfers) < insurance_limit:
                    cycle_cost_100 = calc_collection_cycle_cost_100(num_eggs, blasts)
                    cost_insurance += cycle_cost_100 * 0.3
                    insurance_collections += 1
                else:
                    cost_self += cost_coll_self
                    self_collections += 1
                
            if blasts == 0: continue
            
            if pgta_mode:
                cost_self += blasts * cost_pgta_unit
                available_embryos = np.random.binomial(blasts, euploid_rate)
            else:
                available_embryos = blasts
                
            for _ in range(available_embryos):
                # 移植回数（不成功回数）に応じた動的確率調整
                if pgta_mode:
                    current_lbr = get_adjusted_lbr_pirtea(base_lbr, total_transfers)
                else:
                    current_lbr = get_adjusted_lbr_no_pgta(base_lbr, total_transfers)
                
                current_clin_preg_rate = current_lbr / (1 - misc_rate)

                total_time += 1 
                total_transfers += 1
                
                if pgta_mode:
                    cost_self += cost_trans_self
                    self_transfers += 1
                else:
                    if (past_transfers + insurance_transfers) < insurance_limit:
                        cost_insurance += cost_transfer_insurance
                        insurance_transfers += 1
                    else:
                        cost_self += cost_trans_self
                        self_transfers += 1
                    
                rand_val = np.random.random()
                if rand_val < current_lbr:
                    success = True
                    break
                elif rand_val < current_lbr + (current_clin_preg_rate * misc_rate):
                    total_time += 3
                    cost_insurance += cost_miscarriage
                else:
                    pass
                    
        results.append({
            "time": total_time, 
            "cost_insurance": cost_insurance,
            "cost_self": cost_self,
            "cost_total": cost_insurance + cost_self,
            "collections": total_collections,
            "insurance_collections": insurance_collections,
            "self_collections": self_collections,
            "transfers": total_transfers,
            "insurance_transfers": insurance_transfers,
            "self_transfers": self_transfers,
        })
        
    return pd.DataFrame(results)

def get_default_rates(age, is_pcos):
    if age <= 35: fert = 78
    elif age <= 39: fert = 70
    else: fert = 65

    if age <= 32: blast = 60
    elif age <= 35: blast = 55
    elif age <= 38: blast = 50
    elif age <= 40: blast = 45
    elif age <= 42: blast = 40
    else: blast = 30
    
    if is_pcos:
        fert = max(0, fert - 15) 
        blast = max(0, blast - 2)

    return fert, blast

def get_metrics(df):
    return {
        "median_time": df["time"].quantile(0.5), "p90_time": df["time"].quantile(0.9),
        "mean_coll": df["collections"].mean(), "mean_ins_coll": df["insurance_collections"].mean(), "mean_self_coll": df["self_collections"].mean(),
        "max_coll": df["collections"].quantile(0.9), "p90_ins_coll": df["insurance_collections"].quantile(0.9), "p90_self_coll": df["self_collections"].quantile(0.9),
        "mean_trans": df["transfers"].mean(), "mean_ins_trans": df["insurance_transfers"].mean(), "mean_self_trans": df["self_transfers"].mean(),
        "max_trans": df["transfers"].quantile(0.9), "p90_ins_trans": df["insurance_transfers"].quantile(0.9), "p90_self_trans": df["self_transfers"].quantile(0.9),
        "median_cost_total": df["cost_total"].quantile(0.5), "p90_cost_total": df["cost_total"].quantile(0.9),
        "median_cost_insurance": df["cost_insurance"].quantile(0.5), "median_cost_self": df["cost_self"].quantile(0.5),
        "p90_cost_insurance": df["cost_insurance"].quantile(0.9), "p90_cost_self": df["cost_self"].quantile(0.9),
    }

# --- 4. Streamlit UI設定 ---
st.set_page_config(page_title="不妊治療費用・期間シミュレーター", layout="wide")
st.title("不妊治療の必要費用および期間のシミュレーター")
st.markdown("現在の年齢とAMHから、「保険適用（PGT-Aなし）」と「全額自費（PGT-Aあり）」の期間と費用を比較します。")
st.caption("※本ツールは、Moon et al. (2016)、Pirtea et al. (2020)、およびHFEA統計基準のデータを統合し、反復不成功による着床率の低下（ベイズ更新）を加味して構築しています。")

st.sidebar.header("あなたの情報を入力")
current_age = st.sidebar.slider("現在の年齢", 25, 45, 32)
start_age = st.sidebar.slider("保険での体外受精治療開始時の年齢", 25, 45, 32)
amh = st.sidebar.number_input("AMH (ng/mL)", min_value=0.00, max_value=20.00, value=2.00, step=0.01, format="%.2f")

is_pcos = st.sidebar.checkbox("PCOS（多嚢胞性卵巣症候群）の傾向がある", value=False)
if is_pcos:
    st.sidebar.warning("※PCOSモデル適用中：成熟卵率および正常受精率の低下リスクを補正しています。")

past_transfers = st.sidebar.number_input("これまでに消化した保険移植回数", min_value=0, max_value=6, value=0, step=1)

# 保険リミットの判定（開始時の年齢を基準とする）
ins_limit = 6 if start_age < 40 else (3 if start_age <= 42 else 0)
rem_limit = max(0, ins_limit - past_transfers)
st.sidebar.caption(f"（あなたの保険移植上限: 初回 {ins_limit}回 / 残り: {rem_limit}回）")

# 医学的な予測は現在の年齢を使用
default_eggs = predict_oocytes_moon(current_age, amh)
default_fert, default_blast = get_default_rates(current_age, is_pcos)

st.sidebar.header("詳細パラメータ")
expected_eggs = st.sidebar.number_input("期待採卵個数", min_value=1, max_value=50, value=default_eggs)
fert_rate_input = st.sidebar.number_input("正常受精率 (%)", min_value=0, max_value=100, value=default_fert)
blast_rate_input = st.sidebar.number_input("胚盤胞到達率 (%)", min_value=0, max_value=100, value=default_blast)

st.sidebar.header("費用の設定")
cost_coll_pgta = st.sidebar.number_input("自費での「採卵費用」(円)", value=400000, step=10000)
cost_trans_pgta = st.sidebar.number_input("自費での「胚移植費用」(円)", value=150000, step=10000)
cost_pgta = st.sidebar.number_input("自費での「PGT-A検査代(1個)」(円)", value=100000, step=10000)

df_no_pgta = simulate_ivf(current_age, start_age, expected_eggs, fert_rate_input / 100.0, blast_rate_input / 100.0, False, cost_coll_pgta, cost_trans_pgta, cost_pgta, past_transfers, is_pcos)
df_pgta = simulate_ivf(current_age, start_age, expected_eggs, fert_rate_input / 100.0, blast_rate_input / 100.0, True, cost_coll_pgta, cost_trans_pgta, cost_pgta, past_transfers, is_pcos)

m_no = get_metrics(df_no_pgta)
m_pgta = get_metrics(df_pgta)

# --- 5. 結果の表示（ダッシュボード） ---
st.divider()
st.subheader(f"シミュレーション結論（{current_age}歳 / 期待採卵数 {expected_eggs}個 / 保険残り {rem_limit}回）")

col1, col2 = st.columns(2)
with col1:
    st.info("**🏥 保険適用（PGT-Aなし）の場合**")
    st.metric(label="平均的な治療期間（半数の方が卒業する目安）", value=f"{m_no['median_time']:.1f} ヶ月")
    st.metric(label="必要な総費用（半数の方が卒業する目安）", value=f"{m_no['median_cost_total']/10000:,.0f} 万円")
with col2:
    st.warning("**🔬 全額自費（PGT-Aあり）の場合**")
    st.metric(label="平均的な治療期間（半数の方が卒業する目安）", value=f"{m_pgta['median_time']:.1f} ヶ月")
    st.metric(label="必要な総費用（半数の方が卒業する目安）", value=f"{m_pgta['median_cost_total']/10000:,.0f} 万円")

st.divider()

# --- 詳細データエリア ---
with st.expander("📊 詳細な内訳データを見る（期間・回数・費用）", expanded=True):
    tab_dt_time, tab_dt_count, tab_dt_cost = st.tabs(["⏳ 治療期間", "🏥 採卵・移植回数", "💰 トータル費用"])

    # 1. 治療期間タブ（カード形式）
    with tab_dt_time:
        st.markdown("**■ 平均期間 (半数の方が卒業する目安)**")
        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.markdown("🌱 **保険（PGT-Aなし）**")
                st.markdown(f"<h3 style='margin:0; color:#FB8C00;'>{m_no['median_time']:.1f} <span style='font-size:16px;'>ヶ月</span></h3>", unsafe_allow_html=True)
        with col2:
            with st.container(border=True):
                st.markdown("✨ **PGT-A（全額自費）**")
                st.markdown(f"<h3 style='margin:0; color:#FB8C00;'>{m_pgta['median_time']:.1f} <span style='font-size:16px;'>ヶ月</span></h3>", unsafe_allow_html=True)

        st.markdown("**■ 最長期間 (10人中9人が収まる最大値)**")
        col3, col4 = st.columns(2)
        with col3:
            with st.container(border=True):
                st.markdown("🌱 **保険（PGT-Aなし）**")
                st.markdown(f"<h3 style='margin:0; color:#666;'>{m_no['p90_time']:.1f} <span style='font-size:16px;'>ヶ月</span></h3>", unsafe_allow_html=True)
        with col4:
            with st.container(border=True):
                st.markdown("✨ **PGT-A（全額自費）**")
                st.markdown(f"<h3 style='margin:0; color:#666;'>{m_pgta['p90_time']:.1f} <span style='font-size:16px;'>ヶ月</span></h3>", unsafe_allow_html=True)

    # 2. 採卵・移植回数タブ（ご指定の見やすい余白レイアウト）
    with tab_dt_count:
        st.markdown("**■ 平均的な回数 (半数の方が卒業する目安)**")
        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.markdown("🌱 **保険（PGT-Aなし）**")
                st.markdown(f"""
                <div style='line-height: 1.4; margin-top: 8px;'>
                    <div style='margin-bottom: 12px;'>
                        <span style='font-weight:bold; color:#333;'>採卵:</span> 
                        <span style='font-size:20px; font-weight:bold; color:#43A047;'>{m_no['mean_coll']:.1f}</span> <span style='font-size:14px; color:#333;'>回</span><br>
                        <span style='font-size:12px; color:gray;'>(内訳: 保険 {m_no['mean_ins_coll']:.1f}回 / 自費 {m_no['mean_self_coll']:.1f}回)</span>
                    </div>
                    <div>
                        <span style='font-weight:bold; color:#333;'>移植:</span> 
                        <span style='font-size:20px; font-weight:bold; color:#E53935;'>{m_no['mean_trans']:.1f}</span> <span style='font-size:14px; color:#333;'>回</span><br>
                        <span style='font-size:12px; color:gray;'>(内訳: 保険 {m_no['mean_ins_trans']:.1f}回 / 自費 {m_no['mean_self_trans']:.1f}回)</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
        with col2:
            with st.container(border=True):
                st.markdown("✨ **PGT-A（全額自費）**")
                st.markdown(f"""
                <div style='line-height: 1.4; margin-top: 8px;'>
                    <div style='margin-bottom: 12px;'>
                        <span style='font-weight:bold; color:#333;'>採卵:</span> 
                        <span style='font-size:20px; font-weight:bold; color:#43A047;'>{m_pgta['mean_coll']:.1f}</span> <span style='font-size:14px; color:#333;'>回</span><br>
                        <span style='font-size:12px; color:gray;'>(内訳: 全額自費 {m_pgta['mean_self_coll']:.1f}回)</span>
                    </div>
                    <div>
                        <span style='font-weight:bold; color:#333;'>移植:</span> 
                        <span style='font-size:20px; font-weight:bold; color:#E53935;'>{m_pgta['mean_trans']:.1f}</span> <span style='font-size:14px; color:#333;'>回</span><br>
                        <span style='font-size:12px; color:gray;'>(内訳: 全額自費 {m_pgta['mean_self_trans']:.1f}回)</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("**■ 難航した場合の回数 (10人中9人が収まる最大値)**")
        col3, col4 = st.columns(2)
        with col3:
            with st.container(border=True):
                st.markdown("🌱 **保険（PGT-Aなし）**")
                st.markdown(f"""
                <div style='line-height: 1.4; margin-top: 8px;'>
                    <div style='margin-bottom: 8px;'>
                        <span style='font-weight:bold; color:#555;'>最大採卵:</span> <span style='font-size:16px;'>{m_no['max_coll']:.0f}</span> 回<br>
                        <span style='font-size:12px; color:gray;'>(保険 {m_no['p90_ins_coll']:.0f}回 / 自費 {m_no['p90_self_coll']:.0f}回)</span>
                    </div>
                    <div>
                        <span style='font-weight:bold; color:#555;'>最大移植:</span> <span style='font-size:16px;'>{m_no['max_trans']:.0f}</span> 回<br>
                        <span style='font-size:12px; color:gray;'>(保険 {m_no['p90_ins_trans']:.0f}回 / 自費 {m_no['p90_self_trans']:.0f}回)</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
        with col4:
            with st.container(border=True):
                st.markdown("✨ **PGT-A（全額自費）**")
                st.markdown(f"""
                <div style='line-height: 1.4; margin-top: 8px;'>
                    <div style='margin-bottom: 8px;'>
                        <span style='font-weight:bold; color:#555;'>最大採卵:</span> <span style='font-size:16px;'>{m_pgta['max_coll']:.0f}</span> 回<br>
                        <span style='font-size:12px; color:gray;'>(全額自費 {m_pgta['p90_self_coll']:.0f}回)</span>
                    </div>
                    <div>
                        <span style='font-weight:bold; color:#555;'>最大移植:</span> <span style='font-size:16px;'>{m_pgta['max_trans']:.0f}</span> 回<br>
                        <span style='font-size:12px; color:gray;'>(全額自費 {m_pgta['p90_self_trans']:.0f}回)</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # 3. トータル費用タブ（カード形式）
    with tab_dt_cost:
        st.markdown("**■ 平均総費用 (半数の方が卒業する目安)**")
        c5, c6 = st.columns(2)
        with c5:
            with st.container(border=True):
                st.markdown("🌱 **保険（PGT-Aなし）**")
                st.markdown(f"<h3 style='margin:0; color:#1E88E5;'>{m_no['median_cost_total']/10000:,.0f} <span style='font-size:16px;'>万円</span></h3>", unsafe_allow_html=True)
                st.markdown(f"<span style='font-size:12px;color:gray;'>内訳：保険 約{m_no['median_cost_insurance']/10000:,.0f}万円 ＋ 自費 約{m_no['median_cost_self']/10000:,.0f}万円</span>", unsafe_allow_html=True)
        with c6:
            with st.container(border=True):
                st.markdown("✨ **PGT-A（全額自費）**")
                st.markdown(f"<h3 style='margin:0; color:#E53935;'>{m_pgta['median_cost_total']/10000:,.0f} <span style='font-size:16px;'>万円</span></h3>", unsafe_allow_html=True)
                st.markdown(f"<span style='font-size:12px;color:gray;'>内訳：全額自費</span>", unsafe_allow_html=True)

        st.markdown("**■ 最大総費用 (10人中9人が収まる最大値)**")
        c7, c8 = st.columns(2)
        with c7:
            with st.container(border=True):
                st.markdown("🌱 **保険（PGT-Aなし）**")
                st.markdown(f"<h4 style='margin:0; color:#666;'>{m_no['p90_cost_total']/10000:,.0f} 万円</h4>", unsafe_allow_html=True)
                st.markdown(f"<span style='font-size:12px;color:gray;'>内訳：保険 約{m_no['p90_cost_insurance']/10000:,.0f}万円 ＋ 自費 約{m_no['p90_cost_self']/10000:,.0f}万円</span>", unsafe_allow_html=True)
        with c8:
            with st.container(border=True):
                st.markdown("✨ **PGT-A（全額自費）**")
                st.markdown(f"<h4 style='margin:0; color:#666;'>{m_pgta['p90_cost_total']/10000:,.0f} 万円</h4>", unsafe_allow_html=True)
                st.markdown(f"<span style='font-size:12px;color:gray;'>内訳：全額自費</span>", unsafe_allow_html=True)

# --- グラフ描画 ---
st.divider()
st.subheader("視覚的な比較")
col_g1, col_g2 = st.columns(2)
with col_g1:
    fig_time = go.Figure(data=[
        go.Bar(name='5割(中央値)', x=['保険', '自費'], y=[m_no['median_time'], m_pgta['median_time']], marker_color='#A0AEC0'),
        go.Bar(name='9割(最大値)', x=['保険', '自費'], y=[m_no['p90_time'], m_pgta['p90_time']], marker_color='#4A5568')
    ])
    fig_time.update_layout(title="卒業までの期間(ヶ月)", barmode='group')
    st.plotly_chart(fig_time, use_container_width=True)
with col_g2:
    fig_cost = go.Figure(data=[
        go.Bar(name='5割(中央値)', x=['保険', '自費'], y=[m_no['median_cost_total']/10000, m_pgta['median_cost_total']/10000], marker_color='#90CDF4'),
        go.Bar(name='9割(最大値)', x=['保険', '自費'], y=[m_no['p90_cost_total']/10000, m_pgta['p90_cost_total']/10000], marker_color='#F56565')
    ])
    fig_cost.update_layout(title="卒業までの総費用(万円)", barmode='group')
    st.plotly_chart(fig_cost, use_container_width=True)
