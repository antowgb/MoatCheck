"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AlertTriangle, Database, RefreshCw, Sparkles } from "lucide-react";
import dynamic from "next/dynamic";
import Card from "@/components/Card";
import PageHeader from "@/components/PageHeader";
import { usePortfolios } from "@/components/PortfolioProvider";
import AllocationChangesTable from "@/components/advisor/AllocationChangesTable";
import MacroRegimeCard from "@/components/advisor/MacroRegimeCard";
import RecommendationSummaryCard from "@/components/advisor/RecommendationSummaryCard";
import RiskAndCostImpactPanel from "@/components/advisor/RiskAndCostImpactPanel";
import MetricInfoTooltip from "@/components/explainability/MetricInfoTooltip";
import type { CalculationKind } from "@/components/explainability/CalculationDrawer";
import AdvisorHorizonIntro from "@/components/advisor/AdvisorHorizonIntro";
import PortfolioValuationCard from "@/components/advisor/PortfolioValuationCard";
import PortfolioHoldingsEditor from "@/components/advisor/PortfolioHoldingsEditor";
import SmartDCAPlanner from "@/components/advisor/SmartDCAPlanner";
import PortfolioDiscoveryCard from "@/components/advisor/PortfolioDiscoveryCard";
import { inferPeriodsPerYear } from "@/lib/portfolio-math";
import { api, type AdvisorMarketSnapshot, type AdvisorPolicy, type AdvisorRecommendationResult, type PersistedAdvisorRecommendation, type PortfolioHoldingInput } from "@/lib/api";

const CalculationDrawer = dynamic(() => import("@/components/explainability/CalculationDrawer"), { ssr: false, loading: () => <span className="h-8 w-24 animate-pulse rounded-lg bg-white/[0.04]" aria-hidden="true" /> });
const LongTermGainProjectionCard = dynamic(() => import("@/components/advisor/LongTermGainProjectionCard"), {
  ssr: false,
  loading: () => <Card title="Long-term estimate" className="col-span-12"><div className="h-[340px] animate-pulse rounded-xl bg-white/[0.025]" aria-hidden="true" /></Card>,
});

const inputClass = "quant-control px-3 py-2 text-xs outline-none focus:border-cyan-300/50";
const DEFAULT_POLICY: AdvisorPolicy = {
  holding_period_days: 90,
  minimum_holding_period_days: 30,
  allocation_method: "risk_parity",
  risk_free_rate: 0.03,
  fractional_kelly: 0.2,
  kelly_sizing_enabled: true,
  max_one_way_turnover: 0.1,
  turnover_penalty_factor: 3,
  confidence_level: 0.95,
  max_adv_participation: 0.1,
  regime_lookback_months: 6,
  capital: 1_000_000,
};

function sampleVolatility(columns: number[][]): number[] {
  return columns.map((column) => {
    if (column.length < 2) return 0;
    const mean = column.reduce((sum, value) => sum + value, 0) / column.length;
    return Math.sqrt(column.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (column.length - 1));
  });
}

const periodsPerYear = inferPeriodsPerYear;

function frequencyLabel(periods: number): "daily" | "weekly" | "monthly" | "irregular" {
  if (periods >= 200) return "daily";
  if (periods >= 40) return "weekly";
  if (periods >= 9) return "monthly";
  return "irregular";
}

function dailyEquivalent(values: number[], annualPeriods: number): number[] {
  const scale = Math.sqrt(annualPeriods / 252);
  return values.map((value) => value * scale);
}

function holdingPeriodVaR(snapshot: AdvisorMarketSnapshot, weights: Record<string, number>, horizonDays: number, confidence: number): number {
  const weighted = snapshot.returns.map((row) => row.reduce((sum, value, index) => sum + value * (weights[snapshot.tickers[index]] ?? 0), 0));
  const horizonPeriods = Math.max(1, Math.round(horizonDays * periodsPerYear(snapshot.observation_dates) / 365.25));
  let horizonReturns: number[];
  if (weighted.length > horizonPeriods) {
    horizonReturns = [];
    for (let start = 0; start <= weighted.length - horizonPeriods; start += 1) {
      let growth = 1;
      for (let offset = 0; offset < horizonPeriods; offset += 1) growth *= 1 + weighted[start + offset];
      horizonReturns.push(growth - 1);
    }
  } else {
    horizonReturns = weighted.map((value) => value * Math.sqrt(horizonPeriods));
  }
  if (!horizonReturns.length) return 0;
  horizonReturns.sort((a, b) => a - b);
  const position = (1 - confidence) * (horizonReturns.length - 1);
  const left = Math.floor(position), right = Math.ceil(position);
  const quantile = horizonReturns[left] + (horizonReturns[right] - horizonReturns[left]) * (position - left);
  return Math.max(0, -quantile);
}

function portfolioMoments(recommendation: AdvisorRecommendationResult, weights: Record<string, number>) {
  const tickers = recommendation.tickers;
  const vector = tickers.map((ticker) => weights[ticker] ?? 0);
  const returns = tickers.map((ticker) => recommendation.annualized_expected_returns[ticker] ?? 0);
  const covariance = recommendation.annualized_covariance_matrix;
  const expectedReturn = vector.reduce((sum, weight, i) => sum + weight * returns[i], 0);
  let variance = 0;
  for (let i = 0; i < vector.length; i += 1) {
    for (let j = 0; j < vector.length; j += 1) variance += vector[i] * (covariance[i]?.[j] ?? 0) * vector[j];
  }
  return { expectedReturn, volatility: Math.sqrt(Math.max(0, variance)) };
}

function buildCalculation(kind: CalculationKind, recommendation: AdvisorRecommendationResult, capital: number) {
  const inputs: Record<string, string | number> = {
    "Portfolio assets": recommendation.tickers.join(", "),
    "Investment horizon (days)": recommendation.holding_period_days,
    "Risk confidence": recommendation.confidence_level,
    "Gross turnover": recommendation.gross_turnover,
    "One-way turnover": recommendation.one_way_turnover,
    "Portfolio capital": capital,
  };
  if (kind === "markov-transition") {
    inputs["Regime observations"] = recommendation.regime.observations;
    inputs["Transition matrix"] = recommendation.regime.transition_matrix.map((row) => row.map((x) => x.toFixed(3)).join(" · ")).join(" / ");
  } else {
    inputs["Target weights"] = recommendation.tickers.map((ticker) => `${ticker} ${(recommendation.target_weights[ticker] * 100).toFixed(1)}%`).join(" · ");
    inputs["Annualized expected returns"] = recommendation.tickers.map((ticker) => `${ticker} ${((recommendation.annualized_expected_returns[ticker] ?? 0) * 100).toFixed(1)}%`).join(" · ");
  }
  const steps = kind === "markowitz"
    ? ["Infer the return cadence from observation dates and annualize the sample moments at that frequency.", "Solve a long-only allocation under a fully invested simplex constraint.", "Apply the conservative Kelly sleeve, turnover damping, and maximum one-way turnover cap."]
    : kind === "fractional-kelly"
      ? [`Compute excess returns above the ${((DEFAULT_POLICY.risk_free_rate) * 100).toFixed(1)}% risk-free assumption.`, "Apply the covariance pseudoinverse to obtain a continuous Kelly exposure estimate.", "Scale the exposure by the configured fractional Kelly factor and keep the residual as explicit cash."]
      : kind === "liquidity-var"
        ? ["Estimate the historical loss quantile at the selected confidence level.", "Scale the loss by the maximum ADV-constrained liquidation horizon.", "Add half-spread and square-root market-impact liquidation costs."]
        : ["Calculate a rolling one-month realized-volatility estimate at the observed return frequency.", "Bucket observations into low, normal, and high volatility states.", "Count adjacent state transitions with symmetric smoothing and read the current-state probability row."];
  const caveats = [
    "Historical return moments and transition probabilities are estimates and can change materially across samples.",
    "The turnover penalty is a transparent deterministic overlay; it is not a fitted utility parameter.",
    "Execution impact uses an average daily volume and square-root model, not a live order-book quote.",
  ];
  return { inputs, steps, caveats };
}

export default function AdvisorPage() {
  const { activePortfolio, loading: portfoliosLoading, error: portfolioError, refreshPortfolios } = usePortfolios();
  const [policy, setPolicy] = useState<AdvisorPolicy>(DEFAULT_POLICY);
  const [advByTicker, setAdvByTicker] = useState<Record<string, number>>({});
  const [spreadBpsByTicker, setSpreadBpsByTicker] = useState<Record<string, number>>({});
  const [snapshot, setSnapshot] = useState<AdvisorMarketSnapshot | null>(null);
  const [recommendation, setRecommendation] = useState<AdvisorRecommendationResult | null>(null);
  const [persisted, setPersisted] = useState<PersistedAdvisorRecommendation | null>(null);
  const [currentLvarFraction, setCurrentLvarFraction] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [busyLabel, setBusyLabel] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);
  const [holdings, setHoldings] = useState<PortfolioHoldingInput[]>([]);
  const [monthlyDcaAmount, setMonthlyDcaAmount] = useState(0);
  const [targetHorizonYears, setTargetHorizonYears] = useState(5);
  const [priceCurrencies, setPriceCurrencies] = useState<Record<string, string>>({});
  const [currentPriceByTicker, setCurrentPriceByTicker] = useState<Record<string, number>>({});
  const previewRequest = useRef<AbortController | null>(null);

  const loadMarketSnapshot = useCallback(async (signal: AbortSignal): Promise<AdvisorMarketSnapshot> => {
    if (!activePortfolio) throw new Error("Choose an authenticated saved portfolio first.");
    const tickers = activePortfolio.tickers;
    const [historyByTicker, initialDetails] = await Promise.all([
      api.priceHistories(tickers, signal),
      api.stockDetails(tickers, signal),
    ]);
    const histories = tickers.map((ticker) => historyByTicker[ticker] ?? []);
    const detailsByTicker = new Map(initialDetails.map((detail) => [detail.stock.ticker, detail]));
    const details = tickers.map((ticker) => detailsByTicker.get(ticker)).filter((detail): detail is NonNullable<typeof detail> => Boolean(detail));
    if (details.length !== tickers.length) throw new Error("Some saved assets have no market metadata. Refresh the portfolio data and retry.");
    const missingCurrencies = tickers.filter((ticker, index) => {
      const currency = details[index].stock.currency;
      return typeof currency !== "string" || !/^[A-Za-z]{3}$/.test(currency);
    });
    if (missingCurrencies.length) {
      throw new Error(`Quote currency is not available in the stored asset data for ${missingCurrencies.join(", ")}. Refresh this asset data before running Advisor.`);
    }
    const maps = histories.map((rows) => new Map(rows.filter((row) => Number.isFinite(row.close) && row.close > 0).map((row) => [row.date.slice(0, 10), row.close])));
    const commonDates = [...maps[0].keys()].filter((day) => maps.every((prices) => prices.has(day))).sort();
    const priceDates = commonDates.slice(-253);
    if (priceDates.length < 64) throw new Error("At least 64 aligned price observations are required for a 30+ day advisory.");
    const annualPeriods = periodsPerYear(priceDates);
    const sessionsPerObservation = 252 / annualPeriods;
    const alignedPrices = priceDates.map((day) => maps.map((prices) => prices.get(day) as number));
    const periodReturns = alignedPrices.slice(1).map((current, row) => current.map((price, column) => price / alignedPrices[row][column] - 1));
    const observationDates = priceDates.slice(1);
    const advs = tickers.map((ticker, index) => {
      const bars = details[index].recent_prices.filter((bar) => bar.volume != null && bar.volume > 0 && bar.close > 0).slice(-20);
      const observed = bars.length ? bars.reduce((sum, bar) => sum + bar.close * (bar.volume ?? 0), 0) / bars.length / sessionsPerObservation : 0;
      return advByTicker[ticker] ?? observed;
    });
    const spreads = tickers.map((ticker) => (spreadBpsByTicker[ticker] ?? 10) / 10_000);
    if (advs.some((value) => !(value > 0))) throw new Error("ADV is unavailable from the stored volume history. Enter a positive average daily traded value for every asset below.");
    const latestDate = priceDates[priceDates.length - 1];
    const asOf = new Date(`${latestDate}T23:59:59Z`).toISOString();
    setAdvByTicker(Object.fromEntries(tickers.map((ticker, index) => [ticker, advs[index]])));
    setSpreadBpsByTicker((previous) => Object.fromEntries(tickers.map((ticker) => [ticker, previous[ticker] ?? 10])));
    const currencies = tickers.map((ticker, index) => details[index].stock.currency ?? "");
    const unresolvedCurrencies = tickers.filter((_, index) => !/^[A-Za-z]{3}$/.test(currencies[index]));
    if (unresolvedCurrencies.length) throw new Error(`Quote currency is still missing for ${unresolvedCurrencies.join(", ")}. Check the ticker and refresh its asset data.`);
    if (new Set(currencies.map((code) => code.toUpperCase())).size > 1) {
      const quoteSummary = tickers.map((ticker, index) => `${ticker}=${currencies[index].toUpperCase()}`).join(", ");
      throw new Error(`Advisor found different quote currencies in the saved asset data (${quoteSummary}). Check each asset's stored currency; FX conversion is not available yet.`);
    }
    setPriceCurrencies(Object.fromEntries(tickers.map((ticker, index) => [ticker, currencies[index].toUpperCase()])));
    return {
      as_of: asOf,
      tickers,
      observation_dates: observationDates,
      returns: periodReturns,
      prices: alignedPrices[alignedPrices.length - 1],
      adv_values: advs,
      spreads,
      price_currencies: currencies,
    };
  }, [activePortfolio, advByTicker, spreadBpsByTicker]);

  useEffect(() => () => previewRequest.current?.abort(), [activePortfolio?.id]);

  useEffect(() => {
    let active = true;
    setRecommendation(null);
    setPersisted(null);
    setSnapshot(null);
    setCurrentLvarFraction(null);
    if (!activePortfolio) return () => { active = false; };
    const controller = new AbortController();
    void api.latestAdvisorRecommendation(activePortfolio.id, controller.signal).then((response) => {
      if (active) {
        const savedRecommendation = response.data.recommendation;
        const hasUntrackedStartingPositions = savedRecommendation.tickers.some(
          (ticker) => (savedRecommendation.current_weights[ticker] ?? 0) > 1e-8,
        );
        if (!savedRecommendation.valuation.available && hasUntrackedStartingPositions) {
          setInfo("The saved recommendation used portfolio weights instead of tracked positions. Preview advice again to calculate it from cash and your monthly contribution.");
          return;
        }
        setPersisted(response.data);
        setRecommendation(savedRecommendation);
      }
    }).catch(() => { /* A first-time portfolio has no saved recommendation. */ });
    return () => { active = false; controller.abort(); };
  }, [activePortfolio?.id]);

  useEffect(() => {
    let active = true;
    if (!activePortfolio) return () => { active = false; };
    setCurrentPriceByTicker({});
    setPriceCurrencies({});
    const controller = new AbortController();
    void api.stockDetails(activePortfolio.tickers, controller.signal).then((details) => {
      if (active) {
        setPriceCurrencies(Object.fromEntries(activePortfolio.tickers.map((ticker, index) => [ticker, (details[index].stock.currency ?? "").toUpperCase()])));
        setCurrentPriceByTicker(Object.fromEntries(activePortfolio.tickers.flatMap((ticker, index) => {
          const prices = details[index].recent_prices;
          const latest = prices[prices.length - 1]?.close;
          return Number.isFinite(latest) && latest! > 0 ? [[ticker, latest!]] : [];
        })));
      }
    }).catch(() => { /* Market snapshot validation reports missing price metadata if it matters. */ });
    return () => { active = false; controller.abort(); };
  }, [activePortfolio?.id]);

  useEffect(() => {
    let active = true;
    setHoldings([]);
    setMonthlyDcaAmount(0);
    setTargetHorizonYears(5);
    if (!activePortfolio) return () => { active = false; };
    const controller = new AbortController();
    void api.getPortfolioTracking(activePortfolio.id, controller.signal).then(({ data }) => {
      if (!active) return;
      setHoldings(data.holdings);
      setMonthlyDcaAmount(data.monthly_dca_amount);
      setTargetHorizonYears(data.target_horizon_years);
    }).catch((cause) => {
      if (active) setError(cause instanceof Error ? cause.message : "Could not load your holdings.");
    });
    return () => { active = false; controller.abort(); };
  }, [activePortfolio?.id]);

  async function runPreview(persistFormal: boolean) {
    if (!activePortfolio) return;
    previewRequest.current?.abort();
    const controller = new AbortController();
    previewRequest.current = controller;
    setBusy(true);
    setBusyLabel(persistFormal ? "Calculating & saving" : "Building preview");
    setError(null); setInfo(null);
    try {
      const market = await loadMarketSnapshot(controller.signal);
      if (persistFormal) {
        await api.savePortfolioTrackingSettings(activePortfolio.id, {
          monthly_dca_amount: monthlyDcaAmount,
          target_horizon_years: targetHorizonYears,
        }, controller.signal);
      }
      const result = persistFormal
        ? await api.createAdvisorRecommendation(activePortfolio.id, { market, policy }, controller.signal)
        : await api.previewAdvisor({ market, policy, portfolio_id: activePortfolio.id, monthly_dca_amount: monthlyDcaAmount, target_horizon_years: targetHorizonYears }, controller.signal);
      setSnapshot(market);
      const resultingRecommendation = persistFormal
        ? (result.data as PersistedAdvisorRecommendation).recommendation
        : result.data as AdvisorRecommendationResult;
      const currentWeights = resultingRecommendation.current_weights;
      const valuedCapital = resultingRecommendation.valuation.total_market_value
        ?? (market.tickers.some((ticker) => (currentWeights[ticker] ?? 0) > 1e-8)
          ? policy.capital
          : resultingRecommendation.dca_plan.monthly_amount);
      try {
        if (valuedCapital <= 0) {
          setCurrentLvarFraction(null);
        } else {
          const periodPortfolioReturns = market.returns.map((row) => row.reduce((sum, value, index) => sum + value * (currentWeights[market.tickers[index]] ?? 0), 0));
          const portfolioReturns = dailyEquivalent(periodPortfolioReturns, periodsPerYear(market.observation_dates));
          const periodAssetVols = sampleVolatility(market.tickers.map((_, index) => market.returns.map((row) => row[index])));
          const assetVols = periodAssetVols.map((volatility) => volatility * Math.sqrt(periodsPerYear(market.observation_dates) / 252));
          const liquidity = await api.liquidityAdjustedVar({
            tickers: market.tickers,
            portfolio_returns: portfolioReturns,
            position_values: market.tickers.map((ticker) => (currentWeights[ticker] ?? 0) * valuedCapital),
            advs: market.adv_values,
            volatilities: assetVols,
            spreads: market.spreads,
            confidence_level: policy.confidence_level,
            max_participation_rate: policy.max_adv_participation,
            method: "historical",
          }, controller.signal);
          setCurrentLvarFraction(liquidity.data.liquidity_adjusted_var / valuedCapital);
        }
      } catch {
        setCurrentLvarFraction(null);
      }
      if (persistFormal) {
        const saved = result.data as PersistedAdvisorRecommendation;
        setPersisted(saved);
        setRecommendation(saved.recommendation);
        setInfo(saved.email_notification === "sent"
          ? "Formal recommendation saved for review; a notification email has been sent to your account address."
          : saved.email_notification === "not_configured"
            ? "Formal recommendation saved for review. Email delivery is not configured on the server yet."
            : saved.email_notification === "delivery_failed"
              ? "Formal recommendation saved for review. Email delivery failed; the saved result is still available here."
              : "Formal recommendation saved for human review.");
      } else {
        setPersisted(null);
        setRecommendation(resultingRecommendation);
        setInfo("Unpersisted preview calculated from the latest aligned price snapshot.");
      }
    } catch (cause) {
      if (!controller.signal.aborted) setError(cause instanceof Error ? cause.message : "The advisory calculation could not be completed.");
    } finally {
      if (!controller.signal.aborted) { setBusy(false); setBusyLabel(""); }
    }
  }

  async function saveHoldings(next: PortfolioHoldingInput[]) {
    if (!activePortfolio) return;
    setBusy(true); setError(null); setInfo(null);
    try {
      const response = await api.savePortfolioHoldings(activePortfolio.id, next);
      setHoldings(response.data.holdings);
      setRecommendation(null); setPersisted(null);
      setInfo("Holdings saved. Preview the Advisor to update your valuation and projections.");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not save holdings.");
    } finally { setBusy(false); }
  }

  async function saveTrackingSettings() {
    if (!activePortfolio) return;
    setBusy(true); setError(null); setInfo(null);
    try {
      await api.savePortfolioTrackingSettings(activePortfolio.id, { monthly_dca_amount: monthlyDcaAmount, target_horizon_years: targetHorizonYears });
      setInfo("Monthly contribution plan saved.");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not save your plan.");
    } finally { setBusy(false); }
  }

  async function acknowledge() {
    if (!persisted?.id) return;
    setBusy(true); setError(null);
    try {
      const response = await api.acknowledgeAdvisorRecommendation(persisted.id);
      setPersisted(response.data);
      setInfo("This recommendation is marked as human-reviewed.");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not save review status.");
    } finally { setBusy(false); }
  }

  const derived = useMemo(() => {
    if (!recommendation || !snapshot) return null;
    const current = portfolioMoments(recommendation, recommendation.current_weights);
    const target = portfolioMoments(recommendation, recommendation.target_weights);
    const currentVaR = holdingPeriodVaR(snapshot, recommendation.current_weights, recommendation.holding_period_days, recommendation.confidence_level);
    const currentPortfolioReturns = snapshot.returns.map((row) => row.reduce((sum, value, i) => sum + value * (recommendation.current_weights[snapshot.tickers[i]] ?? 0), 0));
    const vols = sampleVolatility(snapshot.tickers.map((_, i) => snapshot.returns.map((row) => row[i]))).map((volatility) => volatility * Math.sqrt(periodsPerYear(snapshot.observation_dates) / 252));
    return {
      before: {
        ...current,
        var: currentVaR,
        lvar: currentLvarFraction,
      },
      after: {
        ...target,
        var: recommendation.var_loss_fraction,
        lvar: recommendation.liquidity_adjusted_var_loss_fraction,
      },
      currentPortfolioReturns,
      vols,
    };
  }, [recommendation, snapshot, policy.capital, currentLvarFraction]);

  const liveValuation = useMemo(() => {
    const rows = holdings.flatMap((holding) => {
      const price = currentPriceByTicker[holding.ticker];
      if (!Number.isFinite(price) || price <= 0 || priceCurrencies[holding.ticker] !== holding.currency) return [];
      const marketValue = holding.quantity * price;
      const costBasis = holding.quantity * holding.average_buy_price;
      return [{
        ticker: holding.ticker, quantity: holding.quantity, average_buy_price: holding.average_buy_price,
        current_price: price, market_value: marketValue, cost_basis: costBasis,
        unrealized_pnl: marketValue - costBasis, unrealized_pnl_fraction: price / holding.average_buy_price - 1,
        currency: holding.currency,
      }];
    });
    const currencies = new Set(rows.map((row) => row.currency));
    if (!rows.length || currencies.size !== 1 || rows.length !== holdings.length) return undefined;
    const totalMarketValue = rows.reduce((sum, row) => sum + row.market_value, 0);
    const totalCostBasis = rows.reduce((sum, row) => sum + row.cost_basis, 0);
    const unrealized = totalMarketValue - totalCostBasis;
    return {
      available: true, currency: rows[0].currency, total_market_value: totalMarketValue,
      total_cost_basis: totalCostBasis, unrealized_pnl: unrealized,
      unrealized_pnl_fraction: totalCostBasis > 0 ? unrealized / totalCostBasis : null,
      holdings: rows, valuation_basis: "Quantity × latest saved market close. Price freshness follows the data feed.",
    };
  }, [holdings, currentPriceByTicker, priceCurrencies]);

  return <div className="space-y-5">
    <PageHeader title="Portfolio Advisor" subtitle="Review portfolio ideas for plans held over weeks or months. Each suggestion shows the data, estimated costs, and risks behind it." />
    <AdvisorHorizonIntro />

    <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
      <Card title="What you do" className="border-cyan-300/10 bg-cyan-300/[0.025]">
        <p className="text-sm leading-relaxed text-slate-200">Choose your saved portfolio in the top bar. Then preview the advice. If it looks useful, save it for review.</p>
        <p className="mt-2 text-xs leading-relaxed text-slate-500">Previewing does not save anything or send an email. Saving keeps a copy and sends a notification to your account email.</p>
      </Card>
      <Card title="What the Advisor does" className="border-emerald-300/10 bg-emerald-300/[0.025]">
        <p className="text-sm leading-relaxed text-slate-200">It reviews recent price history, estimates a possible allocation, and shows how risk and trading costs may change.</p>
        <p className="mt-2 text-xs leading-relaxed text-slate-500">It is designed for plans held over weeks or months. It does not buy, sell, or place orders.</p>
      </Card>
    </div>

    {!activePortfolio && <Card className="border-amber-300/20 bg-amber-300/[0.04]"><div className="flex gap-3"><AlertTriangle size={17} className="mt-0.5 shrink-0 text-amber-300" /><div><p className="text-sm font-semibold text-amber-100">{portfoliosLoading ? "Loading your portfolios" : portfolioError ? "Sign in to continue" : "Choose a portfolio to begin"}</p><p className="mt-1 text-xs leading-relaxed text-slate-400">{portfolioError ? "Sign in with your MoatCheck email and password using the account control in the top bar. Your saved portfolios are private to your account." : "Select a saved portfolio in the top bar, or create one in My Portfolio."}</p><div className="mt-3 flex flex-wrap items-center gap-4"><button type="button" onClick={() => void refreshPortfolios()} className="inline-flex items-center gap-1.5 text-xs font-medium text-cyan-300"><RefreshCw size={13} />Refresh list</button><a href="/portfolio" className="text-xs font-medium text-emerald-300 hover:text-emerald-200">Open My Portfolio</a></div></div></div></Card>}

    {activePortfolio && <div className="grid grid-cols-12 gap-5">
      <div className="col-span-12"><PortfolioHoldingsEditor tickers={activePortfolio.tickers} holdings={holdings} currencies={priceCurrencies} busy={busy} onSave={(items) => void saveHoldings(items)} /></div>
      <PortfolioValuationCard valuation={liveValuation ?? (recommendation?.valuation.available ? recommendation.valuation : undefined)} />
      <SmartDCAPlanner amount={monthlyDcaAmount} years={targetHorizonYears} recommendation={recommendation} busy={busy} onAmountChange={setMonthlyDcaAmount} onYearsChange={setTargetHorizonYears} onSave={() => void saveTrackingSettings()} />
      <PortfolioDiscoveryCard portfolioId={activePortfolio.id} onApplied={() => void refreshPortfolios()} />
      <Card className="col-span-12">
          <div className="flex flex-col gap-4 xl:flex-row xl:items-center xl:justify-between">
          <div className="flex items-start gap-3"><div className="rounded-lg border border-cyan-300/15 bg-cyan-300/[0.07] p-2.5 text-cyan-200"><Sparkles size={17} /></div><div><p className="text-[10px] font-semibold uppercase tracking-[0.15em] text-cyan-300">Selected portfolio</p><h2 className="mt-1 text-sm font-semibold text-slate-100">{activePortfolio.name}</h2><p className="mt-1 text-xs text-slate-500">{activePortfolio.tickers.join(" · ")} · tracked positions set the starting allocation; no positions means cash</p></div></div>
          <div className="flex flex-wrap items-center gap-2"><button type="button" onClick={() => void runPreview(false)} disabled={busy} className="rounded-lg border border-white/10 px-3.5 py-2.5 text-xs font-semibold text-slate-200 transition hover:border-cyan-300/30 disabled:opacity-50">{busy && busyLabel === "Building preview" ? busyLabel : "1. Preview advice"}</button><button type="button" onClick={() => void runPreview(true)} disabled={busy} className="inline-flex items-center gap-2 rounded-lg bg-emerald-300 px-3.5 py-2.5 text-xs font-semibold text-slate-950 transition hover:bg-emerald-200 disabled:opacity-50">{busy && busyLabel === "Calculating & saving" ? <RefreshCw size={13} className="animate-spin" /> : <Database size={13} />}{busy && busyLabel === "Calculating & saving" ? busyLabel : "2. Save and email advice"}</button></div>
        </div>
        <p className="mt-4 text-xs leading-relaxed text-slate-500">Start with the preview. Saving creates a recommendation for your records and sends an email notification. Neither action places trades.</p>
        <details className="group mt-5 border-t border-white/[0.07] pt-4">
          <summary className="flex cursor-pointer list-none items-center justify-between gap-3 text-xs font-semibold text-slate-300 marker:hidden"><span>Optional settings and trading assumptions</span><span className="text-[10px] font-normal text-slate-500 group-open:hidden">Show</span><span className="hidden text-[10px] font-normal text-slate-500 group-open:inline">Hide</span></summary>
          <p className="mt-2 text-[11px] leading-relaxed text-slate-500">The defaults are ready to use. Change these only if you want to adjust the model assumptions.</p>
        <div className="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-5">
          <label className="block"><span className="flex items-center text-[10px] uppercase tracking-wider text-slate-500">Holding horizon<MetricInfoTooltip label="Holding horizon" description="The return-risk horizon used for medium/long-term recommendation analysis. Values are constrained to 30 to 365 calendar days." /></span><div className="mt-2 flex items-center gap-2"><input type="range" min="30" max="365" step="15" value={policy.holding_period_days} onChange={(event) => setPolicy((p) => ({ ...p, holding_period_days: Number(event.target.value), minimum_holding_period_days: Math.min(p.minimum_holding_period_days, Number(event.target.value)) }))} className="w-full accent-emerald-300" /><span className="w-16 text-right font-mono text-xs text-emerald-200">{policy.holding_period_days}d</span></div></label>
          <label className="block"><span className="flex items-center text-[10px] uppercase tracking-wider text-slate-500">Wait before next change<MetricInfoTooltip label="Minimum interval" description="After a recommendation, the Advisor waits this many days before suggesting another weight change." /></span><select value={policy.minimum_holding_period_days} onChange={(event) => setPolicy((p) => ({ ...p, minimum_holding_period_days: Number(event.target.value) }))} className={`${inputClass} mt-2 w-full`}><option value={30}>30 days</option><option value={60}>60 days</option><option value={90}>90 days</option></select></label>
          <label className="block"><span className="flex items-center text-[10px] uppercase tracking-wider text-slate-500">Allocation approach<MetricInfoTooltip label="Allocation approach" description="Choose how the Advisor balances expected return and risk across your assets." /></span><select value={policy.allocation_method} onChange={(event) => setPolicy((p) => ({ ...p, allocation_method: event.target.value as AdvisorPolicy["allocation_method"] }))} className={`${inputClass} mt-2 w-full`}><option value="risk_parity">Balance risk across assets</option><option value="max_sharpe">Favor return per unit of risk</option><option value="minimum_variance">Lower expected portfolio swings</option></select></label>
          <label className="block"><span className="flex items-center text-[10px] uppercase tracking-wider text-slate-500">Cautious position size<MetricInfoTooltip label="Fractional Kelly" description="This setting uses a small part of the model's suggested investment size to limit the effect of estimation errors." /></span><select value={policy.fractional_kelly} onChange={(event) => setPolicy((p) => ({ ...p, fractional_kelly: Number(event.target.value) }))} className={`${inputClass} mt-2 w-full`}><option value={0.15}>15%</option><option value={0.2}>20%</option><option value={0.25}>25%</option></select></label>
          <label className="block"><span className="flex items-center text-[10px] uppercase tracking-wider text-slate-500">One-way turnover cap<MetricInfoTooltip label="Turnover cap" description="Maximum half-L1 weight change per recommendation. The service also dampens each target move to limit trading frequency." /></span><select value={policy.max_one_way_turnover} onChange={(event) => setPolicy((p) => ({ ...p, max_one_way_turnover: Number(event.target.value) }))} className={`${inputClass} mt-2 w-full`}><option value={0.05}>5%</option><option value={0.1}>10%</option><option value={0.15}>15%</option><option value={0.2}>20%</option></select></label>
        </div>
        <div className="mt-4 grid gap-2 border-t border-white/[0.07] pt-4 sm:grid-cols-2 xl:grid-cols-4">
          {activePortfolio.tickers.map((ticker) => <div key={ticker} className="grid grid-cols-[60px_1fr_1fr] items-center gap-2"><span className="font-mono text-[11px] font-semibold text-slate-300">{ticker}</span><label className="text-[9px] text-slate-500">Avg market value / day ({priceCurrencies[ticker] ?? "quote currency"} m)<input inputMode="decimal" aria-label={`${ticker} average daily traded value in millions of ${priceCurrencies[ticker] ?? "quote currency"}`} className={`${inputClass} mt-1 w-full`} value={advByTicker[ticker] == null ? "" : (advByTicker[ticker] / 1_000_000).toFixed(2)} placeholder="Estimated from volume" onChange={(event) => setAdvByTicker((previous) => ({ ...previous, [ticker]: Number(event.target.value) * 1_000_000 }))} /></label><label className="text-[9px] text-slate-500">Spread (basis points)<input inputMode="decimal" aria-label={`${ticker} full spread basis points`} className={`${inputClass} mt-1 w-full`} value={spreadBpsByTicker[ticker] ?? 10} onChange={(event) => setSpreadBpsByTicker((previous) => ({ ...previous, [ticker]: Number(event.target.value) }))} /></label></div>)}
        </div>
        <p className="mt-3 text-[10px] leading-relaxed text-slate-600">Market value is an estimate of how much of each asset trades in a typical day. It helps estimate how long a position might take to sell. It is not your trade size. Values are shown in millions and rounded for readability. Estimates use recent reported volume; the default spread is 10 basis points. Data cadence: {snapshot ? frequencyLabel(periodsPerYear(snapshot.observation_dates)) : "latest available"}.</p>
        </details>
      </Card>

      {error && <p role="alert" className="col-span-12 rounded-lg border border-rose-300/15 bg-rose-400/[0.05] px-3 py-2.5 text-xs text-rose-200">{error}</p>}
      {info && <p role="status" className="col-span-12 rounded-lg border border-emerald-300/15 bg-emerald-400/[0.05] px-3 py-2.5 text-xs text-emerald-200">{info}</p>}

      {recommendation && <>
        <div className="col-span-12"><RecommendationSummaryCard recommendation={recommendation} persisted={persisted} busy={busy} onAcknowledge={() => void acknowledge()} /></div>
        <div className="col-span-12 flex flex-wrap items-center justify-between gap-3"><div><p className="text-[10px] uppercase tracking-[0.17em] text-cyan-300">How we reached this recommendation</p><p className="mt-1 text-xs text-slate-500">{recommendation.observation_frequency ?? "Older"} price data from {recommendation.snapshot_start_date} to {recommendation.snapshot_end_date}. Input size: {recommendation.input_shape.returns?.join(" by ")} (periods by assets).</p></div><div className="flex flex-wrap gap-3">{(["markowitz", "fractional-kelly", "liquidity-var", "markov-transition"] as CalculationKind[]).map((kind) => { const calculationCapital = recommendation.valuation.total_market_value ?? (recommendation.tickers.some((ticker) => (recommendation.current_weights[ticker] ?? 0) > 1e-8) ? policy.capital : recommendation.dca_plan.monthly_amount); const detail = buildCalculation(kind, recommendation, calculationCapital); const trigger = kind === "markowitz" ? "Allocation" : kind === "fractional-kelly" ? "Position size" : kind === "liquidity-var" ? "Liquidity risk" : "Market outlook"; return <CalculationDrawer key={kind} kind={kind} inputs={detail.inputs} steps={detail.steps} caveats={detail.caveats} triggerLabel={trigger} />; })}</div></div>
        {recommendation.valuation.available
          ? <div className="col-span-12"><AllocationChangesTable recommendation={recommendation} capital={policy.capital} /></div>
          : <Card title="Starting from monthly cash" className="col-span-12">
            <p className="text-xs leading-relaxed text-slate-400">There are no tracked positions to rebalance. The monthly investing plan below shows how your chosen contribution could be allocated; no one-time investment or trades against the saved portfolio weights are assumed.</p>
          </Card>}
        <LongTermGainProjectionCard recommendation={recommendation} />
        <div className="col-span-12 grid grid-cols-12 gap-5">{derived && <div className="col-span-12 xl:col-span-7"><RiskAndCostImpactPanel before={derived.before} after={derived.after} recommendation={recommendation} /></div>}<div className="col-span-12 xl:col-span-5"><MacroRegimeCard recommendation={recommendation} /></div></div>
        <div className="col-span-12 flex flex-wrap items-center justify-between gap-2 rounded-lg border border-white/[0.06] bg-black/10 px-3 py-2 text-[10px] text-slate-600"><span>Model {recommendation.model_version} · {recommendation.observation_frequency ?? "older data"} cadence · risk-free {((policy.risk_free_rate) * 100).toFixed(1)}% · max ADV participation {(policy.max_adv_participation * 100).toFixed(0)}%</span><span>Server calculation {recommendation.latency_ms.toFixed(2)} ms · snapshot {snapshot?.as_of ?? recommendation.snapshot_timestamp}</span></div>
      </>}
      {!recommendation && <Card className="col-span-12 py-12 text-center"><p className="text-sm font-medium text-slate-300">No recommendation loaded</p><p className="mt-1 text-xs text-slate-500">Preview the active portfolio or save a formal recommendation for human review.</p></Card>}
    </div>}

  </div>;
}
