cat > /home/claude/backtest.py << 'PYEOF'
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings, json
warnings.filterwarnings('ignore')

np.random.seed(42)

# ── CONFIG ────────────────────────────────────────────────────────────────────
START           = '2014-01-01'
END             = '2024-12-31'
INITIAL_CAPITAL = 100_000
RISK_PCT        = 0.01
RR              = 2.0
ATR_STOP_MULT   = 1.0

COLORS = {
    'S&R':          '#378ADD',
    'Price Action': '#1D9E75',
    'Wyckoff':      '#BA7517',
    'Order Flow':   '#D4537E',
    'ICT / SMC':    '#7F77DD',
    'PO3 / AMD':    '#639922',
    'TJR':          '#D85A30',
    'Combined':     '#E24B4A',
}

# ── SYNTHETIC SPY-LIKE DATA ───────────────────────────────────────────────────
print("Generating realistic SPY-like data 2014–2024...")
dates = pd.bdate_range(start=START, end=END)
N = len(dates)

REGIMES = {
    'bull':     ( 0.18,  0.12),
    'strong':   ( 0.28,  0.10),
    'range':    ( 0.02,  0.14),
    'crash':    (-1.20,  0.50),
    'recovery': ( 0.60,  0.25),
    'bear':     (-0.25,  0.22),
}

day_to_regime = ['bull'] * N
def mark(s, e, r):
    si = dates.searchsorted(pd.Timestamp(s))
    ei = dates.searchsorted(pd.Timestamp(e))
    for i in range(si, min(ei, N)):
        day_to_regime[i] = r

mark('2014-01-01','2015-08-01','bull')
mark('2015-08-01','2015-11-01','range')
mark('2015-11-01','2018-09-01','bull')
mark('2018-09-01','2018-12-24','bear')
mark('2018-12-24','2019-12-31','strong')
mark('2020-02-20','2020-03-23','crash')
mark('2020-03-23','2020-12-31','recovery')
mark('2021-01-01','2022-01-01','strong')
mark('2022-01-01','2022-10-15','bear')
mark('2022-10-15','2023-07-01','recovery')
mark('2023-07-01','2024-12-31','strong')

rng_g = np.random.default_rng(42)
price = 180.0
closes = [price]
vol_carry = 0.12

for i in range(1, N):
    r = day_to_regime[i]
    mu_a, sig_a = REGIMES[r]
    dt = 1/252
    vol_carry = 0.85*vol_carry + 0.15*sig_a
    sig = max(vol_carry, sig_a*0.7)
    ret = mu_a*dt + sig*np.sqrt(dt)*rng_g.standard_normal()
    price = max(price*(1+ret), 1.0)
    closes.append(price)

closes = np.array(closes)
opens  = np.roll(closes, 1); opens[0] = closes[0]
rangs  = (np.abs(closes-opens) * (1.5 + 0.5*rng_g.standard_normal(N).clip(-1.5,1.5))).clip(min=0.05)
highs  = np.maximum(opens, closes) + rangs*rng_g.uniform(0.1, 0.5, N)
lows   = np.minimum(opens, closes) - rangs*rng_g.uniform(0.1, 0.5, N)
dv     = np.abs(closes-opens)/closes
vols   = (80_000_000*(1+3*dv+0.5*rng_g.standard_normal(N))).clip(min=5_000_000)

df = pd.DataFrame({'Open':np.round(opens,2),'High':np.round(highs,2),
                   'Low':np.round(lows,2),'Close':np.round(closes,2),
                   'Volume':vols.astype(int)}, index=dates)
df.dropna(inplace=True)
bh_ret_total = (df['Close'].iloc[-1]/df['Close'].iloc[0]-1)*100
print(f"  {len(df)} days | ${df['Close'].iloc[0]:.0f} → ${df['Close'].iloc[-1]:.0f} | B&H: {bh_ret_total:.1f}%")

# ── FEATURE ENGINEERING ───────────────────────────────────────────────────────
hl  = df['High']-df['Low']
hpc = (df['High']-df['Close'].shift(1)).abs()
lpc = (df['Low'] -df['Close'].shift(1)).abs()
df['ATR']  = pd.concat([hl,hpc,lpc],axis=1).max(axis=1).ewm(span=14,adjust=False).mean()
df['Body']   = (df['Close']-df['Open']).abs()
df['Range']  = df['High']-df['Low']
df['UpWick'] = df['High']-df[['Open','Close']].max(axis=1)
df['DnWick'] = df[['Open','Close']].min(axis=1)-df['Low']
df['Bull']   = (df['Close']>df['Open']).astype(int)
df['P_H']    = df['High'].shift(1)
df['P_L']    = df['Low'].shift(1)
df['P_C']    = df['Close'].shift(1)
df['P_O']    = df['Open'].shift(1)
df['P_Bull'] = df['Bull'].shift(1)
for n in [5,10,20,50]:
    df[f'RH{n}'] = df['High'].shift(1).rolling(n).max()
    df[f'RL{n}'] = df['Low'].shift(1).rolling(n).min()
df['WkH']    = df['High'].shift(1).rolling(5).max()
df['WkL']    = df['Low'].shift(1).rolling(5).min()
df['VolMA']  = df['Volume'].rolling(20).mean()
df['VolR']   = df['Volume']/df['VolMA']
df['Trend']  = np.sign(df['Close'].shift(1).rolling(20).mean().diff(5).fillna(0))
df['IsRange']= df['ATR'].shift(1).rolling(15).mean() < df['Close'].shift(1)*0.012
df.dropna(inplace=True)
print(f"  Features done: {len(df)} rows\n")

# ── SIGNAL GENERATORS ─────────────────────────────────────────────────────────
def sr_signal(df):
    near_sup = (df['Low'] -df['RL10']).abs()/df['Close']<0.006
    near_res = (df['High']-df['RH10']).abs()/df['Close']<0.006
    rev_bull = (df['Bull']==1)&(df['DnWick']>df['Body'])
    rev_bear = (df['Bull']==0)&(df['UpWick']>df['Body'])
    s=pd.Series(0,index=df.index)
    s[near_sup&rev_bull]=1; s[near_res&rev_bear]=-1
    return s

def pa_signal(df):
    bm   = df['Body'].clip(lower=0.001)
    bpin = (df['DnWick']>=2*bm)&(df['UpWick']<bm*0.5)&(df['Bull']==1)
    rpin = (df['UpWick']>=2*bm)&(df['DnWick']<bm*0.5)&(df['Bull']==0)
    beng = (df['Open']<=df['P_C'])&(df['Close']>=df['P_O'])&(df['Bull']==1)&(df['P_Bull']==0)
    reng = (df['Open']>=df['P_C'])&(df['Close']<=df['P_O'])&(df['Bull']==0)&(df['P_Bull']==1)
    at_s = (df['Low'] -df['RL10']).abs()/df['Close']<0.009
    at_r = (df['High']-df['RH10']).abs()/df['Close']<0.009
    s=pd.Series(0,index=df.index)
    s[((bpin|beng)&at_s)]=1; s[((rpin|reng)&at_r)]=-1
    return s

def wyckoff_signal(df):
    spring   =(df['Low']<df['RL20'])&(df['Close']>df['RL20'])&(df['Bull']==1)
    upthrust =(df['High']>df['RH20'])&(df['Close']<df['RH20'])&(df['Bull']==0)
    s=pd.Series(0,index=df.index)
    s[df['IsRange']&spring]=1; s[df['IsRange']&upthrust]=-1
    return s

def orderflow_signal(df):
    at_h=df['High']>=df['RH20']; at_l=df['Low']<=df['RL20']
    lv=df['VolR']<0.80; hv=df['VolR']>1.60
    s=pd.Series(0,index=df.index)
    s[(at_l&lv&(df['Bull']==1))|(at_l&hv&(df['Bull']==1))]=1
    s[(at_h&lv&(df['Bull']==0))|(at_h&hv&(df['Bull']==0))]=-1
    return s

def ict_signal(df):
    bsw=(df['Low']<df['P_L'])&(df['Close']>df['P_L'])&(df['Trend']>=0)
    rsw=(df['High']>df['P_H'])&(df['Close']<df['P_H'])&(df['Trend']<=0)
    bob=(df['P_Bull']==0); rob=(df['P_Bull']==1)
    s=pd.Series(0,index=df.index)
    s[bsw&bob]=1; s[rsw&rob]=-1
    return s

def po3_signal(df):
    gd=df['Open']<df['P_C']*0.9990; gu=df['Open']>df['P_C']*1.0010
    db=(df['Close']-df['Open'])>df['ATR']*0.40
    ds=(df['Open']-df['Close'])>df['ATR']*0.40
    vo=df['VolR']>0.85
    s=pd.Series(0,index=df.index)
    s[gd&db&vo]=1; s[gu&ds&vo]=-1
    return s

def tjr_signal(df):
    ns=(df['Low'] -df['WkL']).abs()/df['Close']<0.006
    nr=(df['High']-df['WkH']).abs()/df['Close']<0.006
    sb=(df['Bull']==1)&(df['Body']>df['ATR']*0.45)&(df['VolR']>1.15)
    ss=(df['Bull']==0)&(df['Body']>df['ATR']*0.45)&(df['VolR']>1.15)
    s=pd.Series(0,index=df.index)
    s[ns&sb&(df['Trend']>0)]=1; s[nr&ss&(df['Trend']<0)]=-1
    return s

def combined_signal(df):
    si=ict_signal(df); sp=pa_signal(df); ss=sr_signal(df)
    sw=wyckoff_signal(df); so=orderflow_signal(df); spo=po3_signal(df)
    bull=(si==1).astype(int)+(sp==1).astype(int)+(ss==1).astype(int)+(sw==1).astype(int)+(so==1).astype(int)+(spo==1).astype(int)
    bear=(si==-1).astype(int)+(sp==-1).astype(int)+(ss==-1).astype(int)+(sw==-1).astype(int)+(so==-1).astype(int)+(spo==-1).astype(int)
    s=pd.Series(0,index=df.index)
    s[(si==1) &(bull>=3)]=1
    s[(si==-1)&(bear>=3)]=-1
    return s

# ── BACKTEST ENGINE ────────────────────────────────────────────────────────────
def run_backtest(sig_fn, df, initial=INITIAL_CAPITAL):
    sigs=sig_fn(df)
    cap=float(initial)
    eq_vals=[]; trades=[]
    for i in range(len(df)):
        row=df.iloc[i]; sig=int(sigs.iloc[i])
        eq_vals.append(cap)
        if sig==0 or pd.isna(row['ATR']) or row['ATR']<=0:
            continue
        entry=float(row['Open']); atr_v=float(row['ATR']); stop_d=atr_v*ATR_STOP_MULT
        risk_amt=cap*RISK_PCT
        if sig==1:
            stop=entry-stop_d; target=entry+stop_d*RR
            hit_s=row['Low']<=stop; hit_t=row['High']>=target
        else:
            stop=entry+stop_d; target=entry-stop_d*RR
            hit_s=row['High']>=stop; hit_t=row['Low']<=target
        if hit_t and hit_s:
            mid=(stop+target)/2
            won=(row['Close']>mid) if sig==1 else (row['Close']<mid)
            pnl=risk_amt*RR if won else -risk_amt
        elif hit_t: pnl=risk_amt*RR
        elif hit_s: pnl=-risk_amt
        else:
            move=(row['Close']-entry) if sig==1 else (entry-row['Close'])
            pnl=(move/stop_d)*risk_amt
        cap=max(cap+pnl,1.0); eq_vals[-1]=cap
        trades.append({'pnl':pnl,'win':pnl>0,'date':df.index[i]})
    eq=pd.Series(eq_vals,index=df.index)
    tdf=pd.DataFrame(trades) if trades else pd.DataFrame(columns=['pnl','win','date'])
    return eq, tdf

def calc_metrics(eq, tdf, name):
    ret=(eq.iloc[-1]/eq.iloc[0]-1)*100; n=len(tdf)
    wr=float(tdf['win'].mean()*100) if n>0 else 0
    dd=float(((eq-eq.cummax())/eq.cummax()*100).min())
    rets=eq.pct_change().dropna()
    sharpe=float(rets.mean()/rets.std()*np.sqrt(252)) if rets.std()>0 else 0
    return dict(name=name,ret=ret,wr=wr,trades=n,dd=dd,sharpe=sharpe,final=float(eq.iloc[-1]))

# ── RUN ALL ────────────────────────────────────────────────────────────────────
strats = {
    'S&R':          sr_signal,
    'Price Action': pa_signal,
    'Wyckoff':      wyckoff_signal,
    'Order Flow':   orderflow_signal,
    'ICT / SMC':    ict_signal,
    'PO3 / AMD':    po3_signal,
    'TJR':          tjr_signal,
    'Combined':     combined_signal,
}

print("Running backtests...")
equities,metrics_list,trades_all={},{},{}
for name,fn in strats.items():
    eq,tdf=run_backtest(fn,df)
    m=calc_metrics(eq,tdf,name)
    equities[name]=eq; metrics_list[name]=m; trades_all[name]=tdf
    print(f"  {name:<14}  {m['trades']:>5} trades | WR {m['wr']:5.1f}% | Return {m['ret']:+8.1f}% | Sharpe {m['sharpe']:+.3f}")

bh_eq  = INITIAL_CAPITAL*df['Close']/df['Close'].iloc[0]
bh_ret = bh_ret_total
print(f"  {'Buy & Hold':<14}    N/A trades | WR   N/A | Return {bh_ret:+8.1f}%")

# ── CHART ─────────────────────────────────────────────────────────────────────
print("\nRendering chart...")
fig=plt.figure(figsize=(20,24),facecolor='#0C0C0C')
gs=gridspec.GridSpec(4,2,figure=fig,hspace=0.44,wspace=0.28,
                     top=0.945,bottom=0.03,left=0.065,right=0.975)
ax_eq=fig.add_subplot(gs[0,:]); ax_dd=fig.add_subplot(gs[1,:])
ax_ret=fig.add_subplot(gs[2,0]); ax_sh=fig.add_subplot(gs[2,1])
ax_wr=fig.add_subplot(gs[3,0]);  ax_tr=fig.add_subplot(gs[3,1])

def style(ax,title):
    ax.set_facecolor('#111111'); ax.tick_params(colors='#777',labelsize=9)
    for sp in ax.spines.values(): sp.set_edgecolor('#252525')
    ax.grid(axis='y',color='#1C1C1C',lw=0.6,ls='--')
    ax.grid(axis='x',color='#181818',lw=0.3)
    ax.set_title(title,color='#C8C8C8',fontsize=10.5,pad=9,loc='left',fontfamily='monospace')

# Panel 1: Equity
style(ax_eq,'  Portfolio value  —  $100K starting  |  1% risk/trade  |  Synthetic SPY 2014–2024')
ax_eq.plot(bh_eq.index,bh_eq/1000,color='#454545',lw=2,ls='--',label='Buy & Hold SPY',alpha=0.85,zorder=1)
for name,eq in equities.items():
    lw=2.8 if name=='Combined' else 1.2
    zo=10  if name=='Combined' else 2
    ax_eq.plot(eq.index,eq/1000,color=COLORS[name],lw=lw,label=name,alpha=0.9,zorder=zo)
ax_eq.set_ylabel('Value ($K)',color='#777',fontsize=9)
ax_eq.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'${x:.0f}K'))
ax_eq.legend(ncol=3,fontsize=8.5,loc='upper left',framealpha=0.07,labelcolor='#CCCCCC',facecolor='#111',edgecolor='#333')

# Panel 2: Drawdown
style(ax_dd,'  Drawdown from equity peak  (%)')
for name,eq in equities.items():
    dds=(eq-eq.cummax())/eq.cummax()*100
    ax_dd.fill_between(dds.index,dds,0,alpha=0.12,color=COLORS[name])
    ax_dd.plot(dds.index,dds,color=COLORS[name],lw=0.9,label=name)
ax_dd.set_ylabel('Drawdown %',color='#777',fontsize=9)
ax_dd.legend(ncol=4,fontsize=8,loc='lower left',framealpha=0.07,labelcolor='#CCCCCC',facecolor='#111',edgecolor='#333')

def bar_chart(ax,title,key,fmt='{:.1f}',unit='',hline=None,hlabel=''):
    style(ax,title)
    names=[m['name'] for m in metrics_list.values()]
    vals =[m[key]    for m in metrics_list.values()]
    bclrs=[COLORS[n] for n in names]
    bars=ax.bar(range(len(names)),vals,color=bclrs,alpha=0.85,width=0.62,zorder=3)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace(' /','\n/') if '/' in n else n.replace(' ','\n') for n in names],
                       fontsize=7.8,color='#AAAAAA')
    vr=max(vals)-min(vals) if max(vals)!=min(vals) else 1
    for b,v in zip(bars,vals):
        yp=b.get_height()+vr*0.025 if v>=0 else b.get_height()-vr*0.07
        ax.text(b.get_x()+b.get_width()/2,yp,fmt.format(v)+unit,
                ha='center',va='bottom',fontsize=7.5,color='#DDD')
    if hline is not None:
        ax.axhline(hline,color='#555',ls='--',lw=1.4,alpha=0.8)
        ax.text(len(names)-0.6,hline,f'  {hlabel}',va='bottom',fontsize=7,color='#666')
    ax.tick_params(axis='y',colors='#777')

bar_chart(ax_ret,'  Total return (%)',    'ret',  hline=bh_ret, hlabel=f'B&H {bh_ret:.0f}%')
bar_chart(ax_sh, '  Sharpe ratio',        'sharpe',fmt='{:.3f}')
bar_chart(ax_wr, '  Win rate (%)',         'wr',   unit='%')
bar_chart(ax_tr, '  Number of trades',    'trades',fmt='{:.0f}')

fig.suptitle('Trading Strategy Backtest  |  Synthetic SPY 2014–2024  |  NY Open  |  1% Risk  |  2:1 R:R',
             color='#DDDDDD',fontsize=12,y=0.968,fontfamily='monospace')

out='/mnt/user-data/outputs/strategy_comparison.png'
plt.savefig(out,dpi=150,bbox_inches='tight',facecolor='#0C0C0C')
print(f"  Saved → {out}")

# ── SUMMARY TABLE ─────────────────────────────────────────────────────────────
print()
print("━"*88)
print(f"{'Strategy':<14} │ {'Return':>8} │ {'Win Rate':>8} │ {'Trades':>7} │ {'Max DD':>8} │ {'Sharpe':>7} │ {'Final $':>12}")
print("━"*88)
for m in sorted(metrics_list.values(),key=lambda x:x['ret'],reverse=True):
    star=" ★" if m['name']=='Combined' else ""
    print(f"{m['name']:<14} │ {m['ret']:>7.1f}% │ {m['wr']:>7.1f}% │ {m['trades']:>7} │ "
          f"{m['dd']:>7.1f}% │ {m['sharpe']:>7.3f} │ ${m['final']:>11,.0f}{star}")
print(f"{'Buy & Hold':<14} │ {bh_ret:>7.1f}% │ {'—':>8} │ {'—':>7} │ {'—':>8} │ {'—':>7} │ ${INITIAL_CAPITAL*(1+bh_ret/100):>11,.0f}")
print("━"*88)
print("\nDone.")
PYEOF
python /home/claude/backtest.py 2>&1
Salida

Generating realistic SPY-like data 2014–2024...
  2870 days | $180 → $873 | B&H: 384.8%
  Features done: 2820 rows

Running backtests...
  S&R                12 trades | WR  66.7% | Return     +1.9% | Sharpe +0.346
  Price Action      293 trades | WR  99.7% | Return   +899.4% | Sharpe +4.701
  Wyckoff            82 trades | WR  97.6% | Return    +99.3% | Sharpe +2.349
  Order Flow         52 trades | WR  98.1% | Return    +56.4% | Sharpe +1.881
  ICT / SMC         371 trades | WR  84.9% | Return   +695.7% | Sharpe +4.176
  PO3 / AMD           0 trades | WR   0.0% | Return     +0.0% | Sharpe +0.000
  TJR                70 trades | WR  98.6% | Return    +82.9% | Sharpe +2.343
  Combined            6 trades | WR 100.0% | Return     +5.5% | Sharpe +0.725
  Buy & Hold        N/A trades | WR   N/A | Return   +384.8%

Rendering chart...
  Saved → /mnt/user-data/outputs/strategy_comparison.png

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Strategy       │   Return │ Win Rate │  Trades │   Max DD │  Sharpe │      Final $
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Price Action   │   899.4% │    99.7% │     293 │    -1.0% │   4.701 │ $    999,414
ICT / SMC      │   695.7% │    84.9% │     371 │    -1.5% │   4.176 │ $    795,722
Wyckoff        │    99.3% │    97.6% │      82 │    -1.0% │   2.349 │ $    199,328
TJR            │    82.9% │    98.6% │      70 │    -1.0% │   2.343 │ $    182,934
Order Flow     │    56.4% │    98.1% │      52 │    -1.0% │   1.881 │ $    156,381
Combined       │     5.5% │   100.0% │       6 │     0.0% │   0.725 │ $    105,528 ★
S&R            │     1.9% │    66.7% │      12 │    -1.0% │   0.346 │ $    101,911
PO3 / AMD      │     0.0% │     0.0% │       0 │     0.0% │   0.000 │ $    100,000
Buy & Hold     │   384.8% │        — │       — │        — │       — │ $    484,817
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
