"use client";

import { motion, useReducedMotion } from "framer-motion";
import MetricInfoTooltip from "@/components/explainability/MetricInfoTooltip";

const METRIC_HELP: Record<string, string> = {
  "Expected return": "Estimated yearly return based on past data. It is an estimate, not a promise.",
  "Annualized volatility": "How much returns have moved up and down, expressed as a yearly figure.",
  "Sharpe ratio": "A way to compare past returns with the amount of risk taken.",
  "Frontier points": "Number of portfolio options shown in the chart.",
  "Historical VaR · 95%": "A 2% VaR means the model estimates a loss of at least 2% on about 1 in 20 days.",
  "Expected shortfall": "Average loss on the worst days beyond the VaR estimate.",
  "Crisis window return": "Portfolio return during the selected past crisis, before current trading costs.",
  "Worst drawdown": "Largest fall from a previous peak.",
  "Net return": "Change in portfolio value after estimated trading costs.",
  Sortino: "A return measure that focuses on losses rather than all price swings.",
  Calmar: "Yearly return compared with the largest fall in the period.",
  "Max drawdown": "Largest percentage fall from a previous peak during the backtest.",
  "Rebalance friction": "Estimated trading costs from changing portfolio weights.",
  "Order friction estimate": "Estimated spread, market impact, and broker fees for these trades.",
  "Average annual turnover": "How much of the portfolio is bought and sold over a year, on average.",
  "Return during crisis": "Portfolio return during the selected past crisis.",
  "Largest drop": "Largest fall from a previous peak during the selected period.",
  "Rebalancing costs": "Estimated costs from changing portfolio weights.",
  "Estimated cost per trade": "Estimated cost to place the trades in this example.",
  "Average yearly turnover": "How much of the portfolio is bought and sold over a year, on average.",
};

export default function MetricCard({
  label,
  value,
  detail,
  tone = "default",
  info,
  delay = 0,
}: {
  label: string;
  value: string;
  detail?: string;
  tone?: "default" | "positive" | "negative" | "accent";
  info?: string;
  delay?: number;
}) {
  const reduceMotion = useReducedMotion();
  const tones = {
    default: "text-slate-50",
    positive: "text-emerald-300",
    negative: "text-rose-300",
    accent: "text-cyan-300",
  };
  return (
    <motion.div initial={reduceMotion ? false : { opacity: 0, y: 15 }} animate={{ opacity: 1, y: 0 }} transition={reduceMotion ? { duration: 0 } : { duration: 0.42, delay }} className="bento-panel h-full p-5 sm:p-6">
      <p className="flex items-center text-xs font-medium uppercase tracking-[0.1em] text-slate-300">
        {label}
        <MetricInfoTooltip label={label} description={info ?? METRIC_HELP[label] ?? "Historical statistical estimate. Interpret with the stated time horizon, units, and sample coverage; realized future values may differ."} />
      </p>
      <p className={`mt-3 break-words font-mono text-2xl font-semibold tracking-tight tabular-nums sm:text-3xl ${tones[tone]}`}>{value}</p>
      {detail && <p className="mt-2 text-sm leading-relaxed text-slate-400">{detail}</p>}
    </motion.div>
  );
}
