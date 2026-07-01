"""
nq_es_strategy.py  (revised — agent-aware)
==========================================
Core NQ/ES strategy engine.
Now outputs a structured SignalContext consumed by the AI agent,
in addition to the original bar-level DataFrame and backtest trades.

THREE LAYERS OF OUTPUT
──────────────────────
1. run_strategy(es_df, nq_df, params) -> pd.DataFrame
   Bar-by-bar signals, EMAs, correlation, pullback, momentum.
   Used for charting and backtest.

2. build_signal_context(df, params) -> dict
   Structured snapshot of the latest bar.
   Consumed by TraderAgent.decide() to produce a DecisionOutput.

3. backtest(df, params) -> list[dict]
   Historical trade simulation with full entry/exit/SL/TP tracking.
   Used for backtest table and equity curve.

SCM CONFIDENCE SCORING
──────────────────────
The confidence field on SignalContext uses a composite score:
  signal_score 40% | corr_strength 25% | pb_depth 20% | vol_confirm 15%
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

NQ_POINT_VALUE = 20

TF_DEFAULTS: dict[str, dict] = {
    "1min":  dict(pullback_thresh=0.0005, hold_bars=10, sl_pct=0.002,  tp_pct=0.004,
                  corr_window=20, ema_fast=20, ema_slow=50, momentum_bars=3, corr_thresh=0.70),
    "5min":  dict(pullback_thresh=0.001,  hold_bars=8,  sl_pct=0.003,  tp_pct=0.006,
                  corr_window=20, ema_fast=20, ema_slow=50, momentum_bars=3, corr_thresh=0.70),
    "15min": dict(pullback_thresh=0.0015, hold_bars=6,  sl_pct=0.004,  tp_pct=0.008,
                  corr_window=20, ema_fast=20, ema_slow=50, momentum_bars=3, corr_thresh=0.70),
    "30min": dict(pullback_thresh=0.002,  hold_bars=5,  sl_pct=0.005,  tp_pct=0.010,
                  corr_window=20, ema_fast=20, ema_slow=50, momentum_bars=3, corr_thresh=0.70),
    "1hour": dict(pullback_thresh=0.003,  hold_bars=4,  sl_pct=0.007,  tp_pct=0.014,
                  corr_window=20, ema_fast=20, ema_slow=50, momentum_bars=3, corr_thresh=0.70),
    "1day":  dict(pullback_thresh=0.005,  hold_bars=3,  sl_pct=0.010,  tp_pct=0.020,
                  corr_window=5,  ema_fast=5,  ema_slow=8,  momentum_bars=2, corr_thresh=0.50),
}


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
    return df.sort_index()


def run_strategy(es_df: pd.DataFrame, nq_df: pd.DataFrame, params: dict) -> pd.DataFrame:
    es = _prep(es_df)
    nq = _prep(nq_df)
    df = pd.DataFrame(index=nq.index)
    df["NQ_open"]  = nq["open"];  df["NQ_high"] = nq["high"]
    df["NQ_low"]   = nq["low"];   df["NQ_close"] = nq["close"]
    df["NQ_vol"]   = nq["volume"]; df["ES_close"] = es["close"]
    df = df.dropna()

    cw = params["corr_window"]; pt = params["pullback_thresh"]
    ct = params["corr_thresh"];  mb = params["momentum_bars"]

    df["ret_NQ"]   = df["NQ_close"].pct_change()
    df["ret_ES"]   = df["ES_close"].pct_change()
    df["corr_20"]  = df["ret_NQ"].rolling(cw).corr(df["ret_ES"])
    df["ratio"]    = df["NQ_close"] / df["ES_close"]
    df["ratio_ma"] = df["ratio"].rolling(cw).mean()
    df["nq_stronger"] = (df["ratio"] > df["ratio_ma"]).astype(bool)
    df["ema_fast"] = df["NQ_close"].ewm(span=params["ema_fast"]).mean()
    df["ema_slow"] = df["NQ_close"].ewm(span=params["ema_slow"]).mean()
    df["trend_nq"] = np.where(df["ema_fast"] > df["ema_slow"], 1, -1)
    df["pullback"] = (df["NQ_close"] - df["ema_fast"]) / df["ema_fast"]
    df["momentum"] = df["NQ_close"] - df["NQ_close"].shift(mb)
    df["tr"] = pd.concat([
        df["NQ_high"] - df["NQ_low"],
        (df["NQ_high"] - df["NQ_close"].shift()).abs(),
        (df["NQ_low"]  - df["NQ_close"].shift()).abs(),
    ], axis=1).max(axis=1)
    df["atr_14"]   = df["tr"].ewm(span=14).mean()
    df["vol_ratio"] = df["NQ_vol"] / df["NQ_vol"].rolling(20).mean()

    df["long_signal"]  = ((df["trend_nq"]==1)  & (df["pullback"]<-pt) & (df["momentum"]>0)  & (df["nq_stronger"])  & (df["corr_20"]>ct)).astype(bool)
    df["short_signal"] = ((df["trend_nq"]==-1) & (df["pullback"]>pt)  & (df["momentum"]<0)  & (~df["nq_stronger"]) & (df["corr_20"]>ct)).astype(bool)

    df["signal_score"] = (
        (df["trend_nq"]==1).astype(int) + (df["corr_20"]>ct).astype(int) +
        df["nq_stronger"].astype(int)   + (df["pullback"].abs()>pt*0.5).astype(int) +
        (df["momentum"].abs()>0).astype(int)
    )
    df["pb_proximity"] = (df["pullback"].abs() / pt * 100).clip(0, 200).round(1)

    score_norm = df["signal_score"] / 5
    corr_norm  = ((df["corr_20"] - ct) / (1 - ct)).clip(0, 1)
    pb_norm    = (df["pullback"].abs() / (pt * 3)).clip(0, 1)
    vol_norm   = (df["vol_ratio"] / 2).clip(0, 1)
    df["confidence"] = (score_norm*0.40 + corr_norm*0.25 + pb_norm*0.20 + vol_norm*0.15).round(3)

    return df.dropna()


def build_signal_context(df: pd.DataFrame, params: dict, tf: str = "30min") -> dict:
    latest   = df.iloc[-1]
    is_long  = bool(latest["long_signal"])
    is_short = bool(latest["short_signal"])
    score    = int(latest.get("signal_score", 0))
    conf     = float(latest.get("confidence", 0.0))
    signal   = "LONG" if is_long else "SHORT" if is_short else "NONE"
    ep       = float(latest["NQ_close"])
    sl_pct   = params["sl_pct"];  tp_pct = params["tp_pct"]
    sl = round(ep*(1-sl_pct),2) if is_long else round(ep*(1+sl_pct),2)
    tp = round(ep*(1+tp_pct),2) if is_long else round(ep*(1-tp_pct),2)
    ct = params["corr_thresh"]; pt = params["pullback_thresh"]

    return {
        "signal": signal, "score": score, "approaching": not is_long and not is_short and score>=4,
        "confidence": conf, "tf": tf, "timestamp": str(df.index[-1]),
        "entry_price": ep, "sl": sl, "tp": tp, "sl_pct": sl_pct, "tp_pct": tp_pct,
        "hold_bars": params["hold_bars"], "rr_ratio": round(tp_pct/sl_pct,2) if sl_pct>0 else 2.0,
        "risk_usd": round(abs(ep-sl)*NQ_POINT_VALUE,0), "reward_usd": round(abs(ep-tp)*NQ_POINT_VALUE,0),
        "pullback_pct": float(latest["pullback"]), "corr_20": float(latest["corr_20"]),
        "trend": int(latest["trend_nq"]), "atr_14": float(latest.get("atr_14",0)),
        "vol_ratio": float(latest.get("vol_ratio",1)), "nq_close": ep,
        "es_close": float(latest["ES_close"]), "ema_fast": float(latest["ema_fast"]),
        "ema_slow": float(latest["ema_slow"]), "momentum": float(latest["momentum"]),
        "nq_stronger": bool(latest["nq_stronger"]),
        "conditions": [
            {"label": f"Trend EMA{params['ema_fast']}>EMA{params['ema_slow']}", "pass": bool(latest["trend_nq"]==1)},
            {"label": f"Pullback>{pt*100:.2f}% from EMA",                        "pass": bool(abs(latest["pullback"])>pt)},
            {"label": "Momentum direction",                                       "pass": bool(latest["momentum"]!=0)},
            {"label": "NQ outperforming ES",                                      "pass": bool(latest["nq_stronger"])},
            {"label": f"Correlation>{ct}",                                        "pass": bool(latest["corr_20"]>ct)},
        ],
    }


def backtest(df: pd.DataFrame, params: dict) -> list[dict]:
    trades=[]; in_trade=False; trade=None
    for ts, row in df.iterrows():
        if in_trade and trade:
            ep=trade["entry_price"]; d=trade["direction"]; sl=trade["sl"]; tp=trade["tp"]; exited=False
            if d=="LONG":
                if row["NQ_low"]<=sl:   trade.update(exit_price=sl, exit_time=str(ts), exit_reason="SL");   exited=True
                elif row["NQ_high"]>=tp: trade.update(exit_price=tp, exit_time=str(ts), exit_reason="TP");   exited=True
                elif trade["bars_held"]>=params["hold_bars"]: trade.update(exit_price=row["NQ_close"],exit_time=str(ts),exit_reason="TIME"); exited=True
            else:
                if row["NQ_high"]>=sl:  trade.update(exit_price=sl, exit_time=str(ts), exit_reason="SL");   exited=True
                elif row["NQ_low"]<=tp:  trade.update(exit_price=tp, exit_time=str(ts), exit_reason="TP");   exited=True
                elif trade["bars_held"]>=params["hold_bars"]: trade.update(exit_price=row["NQ_close"],exit_time=str(ts),exit_reason="TIME"); exited=True
            if exited:
                xp=trade["exit_price"]; pts=(xp-ep) if d=="LONG" else (ep-xp); r_pts=abs(ep-sl)
                trade["pnl_pts"]=round(pts,2); trade["pnl_usd"]=round(pts*NQ_POINT_VALUE,2)
                trade["r_realised"]=round(pts/r_pts,2) if r_pts>0 else 0
                trades.append(trade); in_trade=False; trade=None
            else:
                trade["bars_held"]+=1; continue
        if not in_trade:
            ep=float(row["NQ_close"])
            if row["long_signal"]:
                trade={"direction":"LONG","entry_price":ep,"entry_time":str(ts),"bars_held":1,
                       "sl":round(ep*(1-params["sl_pct"]),2),"tp":round(ep*(1+params["tp_pct"]),2),
                       "risk_pts":round(ep*params["sl_pct"],2),"reward_pts":round(ep*params["tp_pct"],2),
                       "signal_score":int(row.get("signal_score",0)),"confidence":float(row.get("confidence",0))}
                in_trade=True
            elif row["short_signal"]:
                trade={"direction":"SHORT","entry_price":ep,"entry_time":str(ts),"bars_held":1,
                       "sl":round(ep*(1+params["sl_pct"]),2),"tp":round(ep*(1-params["tp_pct"]),2),
                       "risk_pts":round(ep*params["sl_pct"],2),"reward_pts":round(ep*params["tp_pct"],2),
                       "signal_score":int(row.get("signal_score",0)),"confidence":float(row.get("confidence",0))}
                in_trade=True
    return trades


def performance_summary(trades: list[dict]) -> dict:
    if not trades:
        return {"total_trades":0,"wins":0,"losses":0,"win_rate":0.0,"total_pnl_usd":0.0,
                "avg_win_usd":0.0,"avg_loss_usd":0.0,"profit_factor":0.0,"max_drawdown_usd":0.0,
                "avg_hold_bars":0.0,"exits_by_reason":{},"avg_confidence":0.0}
    pnls=   [t["pnl_usd"] for t in trades]
    wins=   [p for p in pnls if p>0]
    losses= [p for p in pnls if p<=0]
    cum=np.cumsum(pnls); peak=np.maximum.accumulate(cum); dd=float(np.min(cum-peak))
    reasons={}
    for t in trades: r=t.get("exit_reason","?"); reasons[r]=reasons.get(r,0)+1
    return {
        "total_trades":len(trades),"wins":len(wins),"losses":len(losses),
        "win_rate":round(len(wins)/len(trades)*100,1),"total_pnl_usd":round(sum(pnls),2),
        "avg_win_usd":round(sum(wins)/len(wins),2) if wins else 0.0,
        "avg_loss_usd":round(sum(losses)/len(losses),2) if losses else 0.0,
        "profit_factor":round(sum(wins)/abs(sum(losses)),2) if losses else 999.0,
        "max_drawdown_usd":round(dd,2),
        "avg_hold_bars":round(sum(t["bars_held"] for t in trades)/len(trades),1),
        "exits_by_reason":reasons,
        "avg_confidence":round(sum(t.get("confidence",0) for t in trades)/len(trades),3),
    }


if __name__ == "__main__":
    import argparse, json, os
    parser = argparse.ArgumentParser(description="NQ/ES Signal Strategy Processor")
    parser.add_argument("--dir");  parser.add_argument("--nq");  parser.add_argument("--es")
    parser.add_argument("--tf", default="30min", choices=list(TF_DEFAULTS.keys())+["all"])
    parser.add_argument("--out", default=".")
    parser.add_argument("--pullback", type=float); parser.add_argument("--sl", type=float)
    parser.add_argument("--tp", type=float);       parser.add_argument("--hold", type=int)
    parser.add_argument("--corr-thresh", type=float, dest="corr_thresh")
    args = parser.parse_args()
    overrides = {}
    if args.pullback:    overrides["pullback_thresh"]=args.pullback
    if args.sl:          overrides["sl_pct"]=args.sl
    if args.tp:          overrides["tp_pct"]=args.tp
    if args.hold:        overrides["hold_bars"]=args.hold
    if args.corr_thresh: overrides["corr_thresh"]=args.corr_thresh
    os.makedirs(args.out, exist_ok=True)
    tfs = list(TF_DEFAULTS.keys()) if args.tf=="all" else [args.tf]
    for tf in tfs:
        params = {**TF_DEFAULTS.get(tf, TF_DEFAULTS["30min"]), **overrides}
        if args.nq and args.es:
            nq_path, es_path = args.nq, args.es
        elif args.dir:
            d=Path(args.dir)
            cnq=list(d.glob(f"NQ_{tf}*.csv"))+list(d.glob(f"nq_{tf}*.csv"))
            ces=list(d.glob(f"ES_{tf}*.csv"))+list(d.glob(f"es_{tf}*.csv"))
            if not cnq or not ces: print(f"  Skipping {tf}: files not found"); continue
            nq_path=str(cnq[0]); es_path=str(ces[0])
        else:
            parser.error("Provide --dir or --nq and --es"); break
        try:
            df=run_strategy(pd.read_csv(es_path), pd.read_csv(nq_path), params)
            sig=build_signal_context(df, params, tf)
            trades=backtest(df,params); perf=performance_summary(trades)
            ts_str=datetime.now().strftime("%Y%m%d_%H%M%S")
            out_file=Path(args.out)/f"nq_es_{tf}_{ts_str}.json"
            out_file.write_text(json.dumps({"generated_at":datetime.now().isoformat(),"timeframe":tf,
                "params":params,"signal_context":sig,"performance":perf,"trades":trades,
                "bars":df.tail(300).reset_index().assign(timestamp=lambda x:x["timestamp"].astype(str)).round(4).to_dict(orient="records")},indent=2))
            action=sig["signal"] if sig["signal"]!="NONE" else "WAIT"
            print(f"\n{'='*52}\n  {tf.upper()} — {action}  (conf {sig['confidence']:.0%})")
            print(f"  Entry {sig['entry_price']:,.2f}  SL {sig['sl']:,.2f}  TP {sig['tp']:,.2f}")
            print(f"  Trades {perf['total_trades']}  WR {perf['win_rate']}%  P&L ${perf['total_pnl_usd']:,.0f}")
            print(f"  Output: {out_file}\n{'='*52}")
        except Exception as e:
            print(f"  Error on {tf}: {e}"); raise
