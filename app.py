import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# --- 1. 年齢別データベース（25歳〜45歳） ---
db_age_data = {
    25: {"euploid_rate": 0.72, "lbr_no_pgta": 0.50, "misc_no_pgta": 0.12, "lbr_pgta": 0.68, "misc_pgta": 0.060},
    26: {"euploid_rate": 0.71, "lbr_no_pgta": 0.49, "misc_no_pgta": 0.12, "lbr_pgta": 0.68, "misc_pgta": 0.060},
    27: {"euploid_rate": 0.70, "lbr_no_pgta": 0.48, "misc_no_pgta": 0.13, "lbr_pgta": 0.67, "misc_pgta": 0.065},
    28: {"euploid_rate": 0.68, "lbr_no_pgta": 0.47, "misc_no_pgta": 0.14, "lbr_pgta": 0.67, "misc_pgta": 0.065},
    29: {"euploid_rate": 0.66, "lbr_no_pgta": 0.46, "misc_no_pgta": 0.14, "lbr_pgta": 0.66, "misc_pgta": 0.070},
    30: {"euploid_rate": 0.65, "lbr_no_pgta": 0.45, "misc_no_pgta": 0.15, "lbr_pgta": 0.65, "misc_pgta": 0.070},
    31: {"euploid_rate": 0.63, "lbr_no_pgta": 0.44, "misc_no_pgta": 0.16, "lbr_pgta": 0.65, "misc_pgta": 0.070},
    32: {"euploid_rate": 0.61, "lbr_no_pgta": 0.43, "misc_no_pgta": 0.17, "lbr_pgta": 0.64, "misc_pgta": 0.070},
    33: {"euploid_rate": 0.59, "lbr_no_pgta": 0.42, "misc_no_pgta": 0.18, "lbr_pgta": 0.64, "misc_pgta": 0.070},
    34: {"euploid_rate": 0.57, "lbr_no_pgta": 0.41, "misc_no_pgta": 0.19, "lbr_pgta": 0.63, "misc_pgta": 0.075},
    35: {"euploid_rate": 0.55, "lbr_no_pgta": 0.40, "misc_no_pgta": 0.20, "lbr_pgta": 0.63, "misc_pgta": 0.075},
    36: {"euploid_rate": 0.50, "lbr_no_pgta": 0.37, "misc_no_pgta": 0.22, "lbr_pgta": 0.61, "misc_pgta": 0.080},
    37: {"euploid_rate": 0.45, "lbr_no_pgta": 0.33, "misc_no_pgta": 0.25, "lbr_pgta": 0.60, "misc_pgta": 0.080},
    38: {"euploid_rate": 0.40, "lbr_no_pgta": 0.29, "misc_no_pgta": 0.30, "lbr_pgta": 0.60, "misc_pgta": 0.085},
    39: {"euploid_rate": 0.35, "lbr_no_pgta": 0.25, "misc_no_pgta": 0.35, "lbr_pgta": 0.60, "misc_pgta": 0.090},
    40: {"euploid_rate": 0.25, "lbr_no_pgta": 0.20, "misc_no_pgta": 0.40, "lbr_pgta": 0.55, "misc_pgta": 0.100},
    41: {"euploid_rate": 0.20, "lbr_no_pgta": 0.15, "misc_no_pgta": 0.48, "lbr_pgta": 0.55, "misc_pgta": 0.100},
    42: {"euploid_rate": 0.15, "lbr_no_pgta": 0.10, "misc_no_pgta": 0.55, "lbr_pgta": 0.50, "misc_pgta": 0.120},
    43: {"euploid_rate": 0.12, "lbr_no_pgta": 0.07, "misc_no_pgta": 0.60, "lbr_pgta": 0.48, "misc_pgta": 0.130},
    44: {"euploid_rate": 0.10, "lbr_no_pgta": 0.05, "misc_no_pgta": 0.65, "lbr_pgta": 0.45, "misc_pgta": 0.150},
    45: {"euploid_rate": 0.05, "lbr_no_pgta": 0.02, "misc_no_pgta": 0.70, "lbr_pgta": 0.40, "misc_pgta": 0.200},
}

# --- 2. 保険点数ベースの費用計算関数（10割負担額） ---
def calc_collection_cycle_cost_100(eggs, blasts):
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

# --- 3. シミュレーション関数 ---
def simulate_ivf(age, expected_eggs, fert_rate, blast_rate, pgta_mode, cost_coll_self, cost_trans_self, cost_pgta_unit, past_transfers=0, num_trials=1000):
    db = db_age_data[age]
    base_blast_yield = 0.80 * fert_rate * blast_rate * 0.75 
    euploid_rate = db["euploid_rate"]
    
    if age < 40: insurance_limit = 6
    elif age <= 42: insurance_limit = 3
    else: insurance_limit = 0
    
    if pgta_mode:
        lbr = db["lbr_pgta"]
        misc_rate = db["misc_pgta"]
    else:
        lbr = db["lbr_no_pgta"]
        misc_rate = db["misc_no_pgta"]
        
    clin_preg_rate = lbr / (1 - misc_rate)
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
                if rand_val < lbr:
                    success = True
                    break
                elif rand_val < lbr + (clin_preg_rate * misc_rate):
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

def get_default_rates(age):
    if age <= 35: fert = 75
    elif age <= 39: fert = 70
    else: fert = 65
    if age <= 32: blast = 60
    elif age <= 35: blast = 55
    elif age <= 38: blast = 50
    elif age <= 40: blast = 45
    elif age <= 42: blast = 40
    else: blast = 30
    return fert, blast

# --- 4. Streamlit UI設定 ---
st.set_page_config(page_title="PGT-A 費用対効果シミュレーター", layout="wide")
st.title("生児獲得までの道のりシミュレーター")
st.markdown("現在の年齢とAMHから、**「保険適用（PGT-Aなし）」**と**「全額自費（PGT-Aあり）」**の期間と費用を比較します。")

st.sidebar.header("あなたの情報を入力")
age = st.sidebar.slider("年齢", 25, 45, 35)
amh = st.sidebar.number_input("AMH (ng/mL)", min_value=0.1, max_value=10.0, value=2.0, step=0.1)

past_transfers = st.sidebar.number_input("これまでに保険で行った胚移植回数", min_value=0, max_value=6, value=0, step=1)

if age < 40:
    limit_text = "6回"
    remaining_limit = max(0, 6 - past_transfers)
elif age <= 42:
    limit_text = "3回"
    remaining_limit = max(0, 3 - past_transfers)
else:
    limit_text = "0回（適用外）"
    remaining_limit = 0

st.sidebar.caption(f"（あなたの保険移植上限: {limit_text} / 残り: {remaining_limit}回）")

default_eggs = min(25, int(np.ceil(3.5 * amh + 1.5)))
default_fert, default_blast = get_default_rates(age)

st.sidebar.header("詳細パラメータ（実績に合わせて手動変更可）")
expected_eggs = st.sidebar.number_input("期待採卵個数", min_value=1, max_value=50, value=default_eggs)
fert_rate_input = st.sidebar.number_input("正常受精率 (%)", min_value=0, max_value=100, value=default_fert)
blast_rate_input = st.sidebar.number_input("胚盤胞到達率 (%)", min_value=0, max_value=100, value=default_blast)
st.sidebar.caption("※デフォルトでは年齢、AMHから推測される値を自動入力しています。")

st.sidebar.header("費用の設定")
cost_coll_pgta = st.sidebar.number_input("自費での「採卵費用」(円)", value=400000, step=10000)
cost_trans_pgta = st.sidebar.number_input("自費での「胚移植費用」(円)", value=150000, step=10000)
cost_pgta = st.sidebar.number_input("自費での「PGT-A検査代(1個)」(円)", value=100000, step=10000)
st.sidebar.info("※案内文: 「保険適用（PGT-Aなし）」の費用は、標準的な保険点数（3割負担）に基づいて自動計算されます。")

df_no_pgta = simulate_ivf(age, expected_eggs, fert_rate_input / 100.0, blast_rate_input / 100.0, False, cost_coll_pgta, cost_trans_pgta, cost_pgta, past_transfers)
df_pgta = simulate_ivf(age, expected_eggs, fert_rate_input / 100.0, blast_rate_input / 100.0, True, cost_coll_pgta, cost_trans_pgta, cost_pgta, past_transfers)

def get_metrics(df):
    return {
        "p10_time": df["time"].quantile(0.1),
        "mean_coll": df["collections"].mean(),
        "mean_ins_coll": df["insurance_collections"].mean(),
        "mean_self_coll": df["self_collections"].mean(),
        "max_coll": df["collections"].quantile(0.9),
        "p90_ins_coll": df["insurance_collections"].quantile(0.9),
        "p90_self_coll": df["self_collections"].quantile(0.9),
        "mean_trans": df["transfers"].mean(),
        "mean_ins_trans": df["insurance_transfers"].mean(),
        "mean_self_trans": df["self_transfers"].mean(),
        "max_trans": df["transfers"].quantile(0.9),
        "p90_ins_trans": df["insurance_transfers"].quantile(0.9),
        "p90_self_trans": df["self_transfers"].quantile(0.9),
        "median_time": df["time"].quantile(0.5),
        "p90_time": df["time"].quantile(0.9),
        "p10_cost_total": df["cost_total"].quantile(0.1),
        "median_cost_total": df["cost_total"].quantile(0.5),
        "median_cost_insurance": df.sort_values(by="cost_total")["cost_insurance"].quantile(0.5),
        "median_cost_self": df.sort_values(by="cost_total")["cost_self"].quantile(0.5),
        "p90_cost_total": df["cost_total"].quantile(0.9),
        "p90_cost_insurance": df.sort_values(by="cost_total")["cost_insurance"].quantile(0.9),
        "p90_cost_self": df.sort_values(by="cost_total")["cost_self"].quantile(0.9),
    }

m_no = get_metrics(df_no_pgta)
m_pgta = get_metrics(df_pgta)

# --- 6. 結果の表示 ---
st.divider()
st.subheader(f"シミュレーション詳細（{age}歳 / 期待採卵数 {expected_eggs}個 / 保険残り {remaining_limit}回）")

# サマリーボックス（レイアウト改善版）
st.success(f"""
**💡 あなたのシミュレーション結果** （目安レンジ：順調なケース(上位10%) 〜 難航するケース(90%が収まる範囲)）

**【保険適用（PGT-Aなし）の場合】**
* ⏳ 不妊治療期間： **{m_no['p10_time']:.0f}ヶ月 〜 {m_no['p90_time']:.0f}ヶ月** （中央値 {m_no['median_time']:.1f}ヶ月）
* 💰 必要な総費用： **{m_no['p10_cost_total']/10000:.0f}万円 〜 {m_no['p90_cost_total']/10000:.0f}万円** （中央値 {m_no['median_cost_total']/10000:.0f}万円）

---
**【全額自費（PGT-Aあり）の場合】**
* ⏳ 不妊治療期間： **{m_pgta['p10_time']:.0f}ヶ月 〜 {m_pgta['p90_time']:.0f}ヶ月** （中央値 {m_pgta['median_time']:.1f}ヶ月）
* 💰 必要な総費用： **{m_pgta['p10_cost_total']/10000:.0f}万円 〜 {m_pgta['p90_cost_total']/10000:.0f}万円** （中央値 {m_pgta['median_cost_total']/10000:.0f}万円）
""")

if m_pgta['median_time'] > m_no['median_time'] and age <= 43:
    st.warning(f"""
    **🩺 【重要な解説】なぜPGT-A（全額自費）の方が、治療期間が長くなっているの？**
    
    結果を見て、「高い自費診療のPGT-Aをしているのに、なぜ期間が長くなっているの？」と驚かれたかもしれません。
    これはバグではなく、現在の設定年齢（{age}歳）とAMHから推測される**「正常胚の獲得率」**に基づくリアルな現実（統計結果）です。
    
    PGT-Aは流産による心身の負担やロスタイムを防ぐ素晴らしい技術です。しかし、厳格な検査を行うため、**「移植の基準を満たす正常胚」が見つかるまで、何度も何度も採卵を繰り返さなければならない**という側面があります。
    
    現在のあなたのシミュレーションでは、流産を回避して節約できる時間よりも、**「正常胚に出会うまでの長い採卵期間」の方が上回ってしまっている**状態です。
    
    このデータは、**「PGT-Aのために採卵を繰り返すよりも、獲得できた胚盤胞をどんどん移植していく方が、結果的に早く赤ちゃんに出会える可能性が高い」**という強力な示唆を与えてくれます。ぜひ、この結果も踏まえて治療方針をご検討ください。
    """)

if m_no['p90_self_trans'] > 0:
    st.error(f"⚠️ **【重要】保険適用回数の上限到達リスクがあります**")
    st.markdown(f"現在、あなたの設定における残りの保険適用移植回数は**{remaining_limit}回**です。胚移植の不成功が続いた場合、9割の方の中には**「保険適用の移植回数上限を超え、途中から全額自費での採卵・移植に切り替わる」**リスクがあります（以下の「費用」の表を参照）。")

st.markdown("#### ⏳ 治療期間の目安")
df_time = pd.DataFrame({
    "項目": [
        "**平均的な不妊治療期間** (5割の人が卒業できる期間)",
        "**最長の不妊治療期間** (9割の人が卒業できる期間)"
    ],
    "保険（PGT-Aなし）": [
        f"{m_no['median_time']:.1f} ヶ月",
        f"{m_no['p90_time']:.1f} ヶ月"
    ],
    "PGT-A（全額自費）": [
        f"{m_pgta['median_time']:.1f} ヶ月",
        f"{m_pgta['p90_time']:.1f} ヶ月"
    ]
})
st.table(df_time.set_index("項目"))

st.markdown("#### 🏥 採卵・移植回数の内訳")
df_count = pd.DataFrame({
    "項目": [
        "**平均採卵回数**",
        "　└ うち保険適用回数",
        "　└ うち全額自費回数",
        "**最大採卵回数** (9割の人がこの回数以内に収まる)",
        "　└ うち保険適用回数",
        "　└ うち全額自費回数",
        "**平均胚移植回数** (卒業までに平均何回移植するか)",
        "　└ うち保険適用回数",
        "　└ うち全額自費回数",
        "**最大胚移植回数** (9割の人がこの回数以内に収まる)",
        "　└ うち保険適用回数",
        "　└ うち全額自費回数",
    ],
    "保険（PGT-Aなし）": [
        f"{m_no['mean_coll']:.1f} 回",
        f"{m_no['mean_ins_coll']:.1f} 回",
        f"{m_no['mean_self_coll']:.1f} 回",
        f"{m_no['max_coll']:.0f} 回",
        f"{m_no['p90_ins_coll']:.0f} 回",
        f"{m_no['p90_self_coll']:.0f} 回",
        f"{m_no['mean_trans']:.1f} 回",
        f"{m_no['mean_ins_trans']:.1f} 回",
        f"{m_no['mean_self_trans']:.1f} 回",
        f"{m_no['max_trans']:.0f} 回",
        f"{m_no['p90_ins_trans']:.0f} 回",
        f"{m_no['p90_self_trans']:.0f} 回",
    ],
    "PGT-A（全額自費）": [
        f"{m_pgta['mean_coll']:.1f} 回",
        "0 回",
        f"{m_pgta['mean_self_coll']:.1f} 回",
        f"{m_pgta['max_coll']:.0f} 回",
        "0 回",
        f"{m_pgta['p90_self_coll']:.0f} 回",
        f"{m_pgta['mean_trans']:.1f} 回",
        "0 回",
        f"{m_pgta['mean_self_trans']:.1f} 回",
        f"{m_pgta['max_trans']:.0f} 回",
        "0 回",
        f"{m_pgta['p90_self_trans']:.0f} 回",
    ]
})
st.table(df_count.set_index("項目"))

st.markdown("#### 💰 トータル費用の目安")
df_cost = pd.DataFrame({
    "項目": [
        "**平均総費用** (5割の人がこの値段以内に収まる)",
        "　└ 保険適用範囲内の負担額",
        "　└ 保険上限超過による自費分 (または全額自費)",
        "**最大総費用** (9割の人がこの値段以内に収まる)",
        "　└ 保険適用範囲内の負担額",
        "　└ 保険上限超過による自費分 (または全額自費)",
    ],
    "保険（PGT-Aなし）": [
        f"**{m_no['median_cost_total']/10000:,.0f} 万円**",
        f"{m_no['median_cost_insurance']/10000:,.0f} 万円",
        f"{m_no['median_cost_self']/10000:,.0f} 万円",
        f"**{m_no['p90_cost_total']/10000:,.0f} 万円**",
        f"{m_no['p90_cost_insurance']/10000:,.0f} 万円",
        f"{m_no['p90_cost_self']/10000:,.0f} 万円",
    ],
    "PGT-A（全額自費）": [
        f"**{m_pgta['median_cost_total']/10000:,.0f} 万円**",
        "0 万円",
        f"{m_pgta['median_cost_total']/10000:,.0f} 万円",
        f"**{m_pgta['p90_cost_total']/10000:,.0f} 万円**",
        "0 万円",
        f"{m_pgta['p90_cost_total']/10000:,.0f} 万円",
    ]
})
st.table(df_cost.set_index("項目"))

# --- 7. グラフの描画 ---
st.divider()
st.subheader("視覚的な比較")

col_g1, col_g2 = st.columns(2)

with col_g1:
    fig_time = go.Figure(data=[
        go.Bar(name='5割の人が卒業', x=['保険(PGT-Aなし)', '自費(PGT-Aあり)'], y=[m_no['median_time'], m_pgta['median_time']], marker_color='#A0AEC0'),
        go.Bar(name='9割の人が卒業', x=['保険(PGT-Aなし)', '自費(PGT-Aあり)'], y=[m_no['p90_time'], m_pgta['p90_time']], marker_color='#4A5568')
    ])
    fig_time.update_layout(title="卒業までに必要な期間（ヶ月）", barmode='group')
    st.plotly_chart(fig_time, use_container_width=True)

with col_g2:
    col_g2a, col_g2b = st.columns(2)
    with col_g2a:
        trace_ins_mean = go.Bar(name='保険適用内負担分', x=['保険(PGT-Aなし)', '自費(PGT-Aあり)'], y=[m_no['median_cost_insurance']/10000, 0], marker_color='#90CDF4')
        trace_self_mean = go.Bar(name='自費診療分', x=['保険(PGT-Aなし)', '自費(PGT-Aあり)'], y=[m_no['median_cost_self']/10000, m_pgta['median_cost_total']/10000], marker_color='#F56565')
        fig_cost_median = go.Figure(data=[trace_ins_mean, trace_self_mean])
        fig_cost_median.update_layout(title="平均的な総費用(中央値)", barmode='stack', yaxis_title="万円", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_cost_median, use_container_width=True)
        
    with col_g2b:
        trace_p90_ins = go.Bar(name='保険適用内負担分', x=['保険(PGT-Aなし)', '自費(PGT-Aあり)'], y=[m_no['p90_cost_insurance']/10000, 0], marker_color='#A3BFD9')
        trace_p90_self = go.Bar(name='自費診療分', x=['保険(PGT-Aなし)', '自費(PGT-Aあり)'], y=[m_no['p90_cost_self']/10000, m_pgta['p90_cost_total']/10000], marker_color='#C53030')
        fig_cost_p90 = go.Figure(data=[trace_p90_ins, trace_p90_self])
        fig_cost_p90.update_layout(title="最大総費用(P90)の比較", barmode='stack', yaxis_title="万円", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_cost_p90, use_container_width=True)