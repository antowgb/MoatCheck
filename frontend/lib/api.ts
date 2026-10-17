// Empty by default: the frontend is served by the backend, /api calls are
// same-origin. In dev (next dev on :3000), set NEXT_PUBLIC_API_URL=http://localhost:8000.
const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "";

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// The backend runs on a free-tier host that spins down after inactivity and
// takes tens of seconds to wake up: the first request(s) after idle can hit a
// transient 5xx (or fail at the network level) while it's still starting.
// Retrying a couple of times, only for read (GET) requests, smooths over
// that cold start instead of surfacing it as a hard failure the user has to
// manually retry (e.g. by hard-refreshing the page).
const RETRYABLE_STATUS = new Set([429, 500, 502, 503, 504]);
const MAX_ATTEMPTS = 3;
const RETRY_DELAY_MS = 1200;

export interface ApiResponseMeta {
  execution_time_ms: number;
  input_shape: Record<string, number | number[]>;
  snapshot_timestamp?: string | null;
}

export interface ApiEnvelope<T> {
  data: T;
  meta: ApiResponseMeta;
}

export interface SavedPortfolio {
  id: string;
  owner_id: string;
  name: string;
  tickers: string[];
  target_weights: Record<string, number>;
  risk_constraints: Record<string, unknown>;
  optimization_parameters: Record<string, unknown>;
  description: string | null;
  updated_at: string;
}

export interface AdvisorMarketSnapshot {
  as_of: string;
  tickers: string[];
  observation_dates: string[];
  returns: number[][];
  prices: number[];
  adv_values: number[];
  spreads: number[];
  price_currencies?: string[];
}

export interface PortfolioHoldingInput {
  ticker: string;
  quantity: number;
  average_buy_price: number;
  currency: string;
  acquired_at: string;
}

export interface PortfolioTrackingData {
  holdings: PortfolioHoldingInput[];
  monthly_dca_amount: number;
  target_horizon_years: number;
}

export interface AdvisorPolicy {
  holding_period_days: number;
  minimum_holding_period_days: number;
  allocation_method: "risk_parity" | "max_sharpe" | "minimum_variance";
  risk_free_rate: number;
  fractional_kelly: number;
  kelly_sizing_enabled: boolean;
  max_one_way_turnover: number;
  turnover_penalty_factor: number;
  confidence_level: number;
  max_adv_participation: number;
  regime_lookback_months: number;
  capital: number;
}

export interface AdvisorPreviewRequest {
  market: AdvisorMarketSnapshot;
  policy?: Partial<AdvisorPolicy>;
  portfolio_id?: string;
  portfolio_name?: string;
  tickers?: string[];
  current_weights?: Record<string, number>;
  monthly_dca_amount?: number;
  target_horizon_years?: number;
}

export interface AdvisorRationaleStep {
  code: string;
  title: string;
  explanation: string;
  evidence: Record<string, unknown>;
}

export interface AdvisorRecommendationResult {
  id?: string;
  owner_id?: string;
  portfolio_id?: string;
  status?: "pending_review" | "acknowledged";
  acknowledged_at?: string | null;
  horizon_days?: number;
  portfolio_name: string;
  tickers: string[];
  current_weights: Record<string, number>;
  target_weights: Record<string, number>;
  weight_deltas: Record<string, number>;
  allocation_method: AdvisorPolicy["allocation_method"];
  annualized_expected_returns: Record<string, number>;
  annualized_covariance_matrix: number[][];
  expected_holding_period_return_fraction: number;
  var_loss_fraction: number;
  cvar_loss_fraction: number;
  confidence_level: number;
  holding_period_days: number;
  observation_frequency?: "daily" | "weekly" | "monthly" | "irregular";
  periods_per_year?: number;
  confidence_score: number;
  confidence_interpretation: string;
  model_version: string;
  holding_period_return_method: string;
  liquidity_adjusted_var_currency: number;
  liquidity_adjusted_var_loss_fraction: number;
  liquidation_horizon_days: number;
  execution_cost: {
    total_execution_drag: number;
    total_traded_notional: number;
    total_friction_bps: number;
    cost_breakdown: Record<string, number>;
    n_orders: number;
    n_active_orders: number;
    [key: string]: unknown;
  };
  gross_turnover: number;
  one_way_turnover: number;
  estimated_annualized_turnover_drag_fraction: number;
  regime: {
    labels: string[];
    current_state: string;
    probability_next_period: Record<string, number>;
    stationary_distribution: Record<string, number>;
    transition_matrix: number[][];
    lookback_months: number;
    observations: number;
    high_volatility_transition_probability: number;
  };
  martingale_diagnostics: Record<string, unknown>;
  rationale: AdvisorRationaleStep[];
  snapshot_timestamp: string;
  snapshot_start_date: string;
  snapshot_end_date: string;
  latency_ms: number;
  input_shape: Record<string, number[]>;
  model_conventions: Record<string, string>;
  valuation: {
    available: boolean;
    currency: string;
    total_market_value: number | null;
    total_cost_basis: number | null;
    unrealized_pnl: number | null;
    unrealized_pnl_fraction: number | null;
    holdings: {
      ticker: string; quantity: number; average_buy_price: number; current_price: number;
      market_value: number; cost_basis: number; unrealized_pnl: number;
      unrealized_pnl_fraction: number; currency: string;
    }[];
    valuation_basis: string;
  };
  dca_plan: {
    monthly_amount: number;
    currency: string;
    orders: { ticker: string; side: "buy"; amount: number; share_of_monthly_dca: number; estimated_quantity: number }[];
    buy_only: boolean;
    unallocated_amount: number;
    basis: string;
  };
  projections: {
    annualized_expected_return: number;
    annualized_volatility: number;
    initial_capital: number;
    monthly_dca_amount: number;
    target_horizon_years: number;
    scenario_labels: Record<string, string>;
    method: string;
    currency: string;
    years: {
      year: number; total_capital_deposited: number;
      pessimistic_value: number; median_value: number; optimistic_value: number;
      pessimistic_net_gain: number; median_net_gain: number; optimistic_net_gain: number;
    }[];
  };
}

export interface DiscoveryPortfolioMetrics {
  expected_return: number | null;
  volatility: number | null;
  sharpe_ratio: number | null;
  weights?: Record<string, number>;
  long_term_projection_per_unit?: { years: number; p10_value_per_unit: number; p50_value_per_unit: number; p90_value_per_unit: number }[];
}

export interface DiscoverySwapProposal {
  proposal_id: string;
  swap_count: number;
  swaps: { replace_ticker: string; with_ticker: string; asset_sharpe_before: number; asset_sharpe_after: number }[];
  target_weights: Record<string, number>;
  expected_return: number;
  volatility: number;
  sharpe_ratio: number;
}

export interface PortfolioDiscoveryData {
  portfolio_id: string;
  portfolio_name: string;
  as_of: string;
  lookback_years: number;
  observations: number;
  universe_assets_evaluated: number;
  current_portfolio: DiscoveryPortfolioMetrics;
  replacement_candidates: Record<string, unknown>[];
  swap_proposals: DiscoverySwapProposal[];
  model_portfolios: { maximum_sharpe: DiscoveryPortfolioMetrics; minimum_risk: DiscoveryPortfolioMetrics };
  methodology: Record<string, string | number>;
  telemetry: { execution_time_ms: number; input_shape: Record<string, number[]> };
}

export interface PersistedAdvisorRecommendation {
  id: string;
  owner_id: string;
  portfolio_id: string;
  horizon_days: number;
  minimum_holding_period_days: number;
  snapshot_timestamp: string;
  status: "pending_review" | "acknowledged";
  acknowledged_at: string | null;
  recommendation: AdvisorRecommendationResult;
  rationale: AdvisorRationaleStep[];
  telemetry: Record<string, unknown>;
  created_at: string;
  email_notification?: "sent" | "not_requested" | "not_configured" | "delivery_failed" | "skipped_no_account_email";
}

export interface QuantFrontierPoint {
  weights: Record<string, number>;
  assets?: string[];
  expected_return: number | null;
  variance?: number;
  volatility: number;
  sharpe_ratio: number | null;
  success: boolean;
  message?: string;
  iterations?: number;
  latency_ms?: number;
}

export interface QuantPortfolioRequest {
  tickers: string[];
  expected_returns: number[];
  covariance_matrix: number[][];
  risk_free_rate?: number;
  points?: number;
}

export interface PortfolioOptimizationData {
  weights?: Record<string, number>;
  assets?: string[];
  expected_return?: number | null;
  volatility?: number;
  sharpe_ratio?: number | null;
  frontier?: QuantFrontierPoint[];
  [key: string]: unknown;
}

export interface PortfolioRiskRequest {
  tickers: string[];
  weights: number[];
  returns: number[][];
  confidence_level?: number;
  method?: "historical" | "parametric";
}

export interface PortfolioRiskData {
  var: number;
  cvar: number;
  confidence: number;
  method: string;
  observations: number;
}

export interface LiquidityVaRRequest {
  tickers: string[];
  portfolio_returns: number[];
  position_values: number[];
  advs: number[];
  volatilities: number[];
  spreads: number[];
  confidence_level?: number;
  max_participation_rate?: number;
  method?: "historical" | "parametric";
}

export interface LiquidityVaRData {
  liquidity_adjusted_var: number;
  standard_var: number;
  daily_var_fraction: number;
  horizon_scaled_var: number;
  max_liquidation_days: number;
  liquidation_days_by_position: number[];
  spread_liquidation_cost: number;
  market_impact_liquidation_cost: number;
  total_liquidation_cost: number;
  method: string;
  confidence_level: number;
  total_position_value: number;
}

export interface HistoricalStressRequest {
  tickers: string[];
  weights: number[];
  dates: string[];
  asset_returns: number[][];
  scenario_start: string;
  scenario_end: string;
}

export interface HistoricalStressData {
  scenario_start: string;
  scenario_end: string;
  n_observations: number;
  cumulative_return: number;
  worst_drawdown: number;
  peak_date: string;
  trough_date: string;
  drawdown_path: Record<string, number>;
  portfolio_return_path: Record<string, number>;
  asset_contributions: Record<string, number>;
}

export interface CrisisScenarioAssetMetadata {
  label: string;
  symbol: string;
  field: string;
  type: string;
  source_url: string;
  note?: string;
}

export interface CrisisScenarioCatalogEntry {
  id: string;
  title: string;
  start_date: string;
  end_date: string;
  base_close_date: string;
  observations: number;
  assets: string[];
  asset_metadata: Record<string, CrisisScenarioAssetMetadata>;
  source: string;
  retrieved_on: string;
  calendar_note: string;
  limitations: string[];
}

export interface HistoricalCrisisReplayData {
  scenario: CrisisScenarioCatalogEntry;
  tickers: string[];
  weights: number[];
  replay: HistoricalStressData;
  daily_returns: { date: string; returns: Record<string, number> }[];
  rolling_volatility_path: {
    date: string;
    portfolio_annualized_volatility: number | null;
    asset_annualized_volatility: Record<string, number | null>;
  }[];
}

export interface RebalanceSimulationRequest {
  tickers: string[];
  initial_weights: number[];
  target_weights: number[][];
  prices: number[][];
  initial_wealth?: number;
  execution_cost_bps?: number;
  max_turnover?: number | null;
  periods_per_year?: number;
}

export interface RebalanceSimulationData {
  wealth_path: Record<string, number>;
  cash_flows: Record<string, number | string>[];
  turnover_series: Record<string, number>;
  total_turnover: number;
  average_annual_turnover: number;
  total_transaction_cost: number;
  total_transaction_drag_bps: number;
  execution_cost_bps_of_traded_notional: number;
}

export interface ExecutionCostRequest {
  trade_sizes: number[];
  prices: number[];
  advs: number[];
  volatilities: number[];
  spreads: number[];
  fixed_fee_per_trade?: number;
  pct_broker_fee?: number;
}

export interface ExecutionCostData {
  total_execution_drag: number;
  total_traded_notional: number;
  total_friction_bps: number;
  cost_breakdown: Record<string, number>;
  per_order_costs: number[];
  n_orders: number;
  n_active_orders: number;
}

async function requestHealth(): Promise<{ status: string; latencyMs: number }> {
  const started = performance.now();
  const res = await fetch(`${API_URL}/health`, { cache: "no-store", signal: AbortSignal.timeout(10_000) });
  if (!res.ok) throw new Error(`API health ${res.status}`);
  const body = (await res.json()) as { status: string };
  const latencyMs = performance.now() - started;
  publishTelemetry({ roundTripLatencyMs: latencyMs });
  return { status: body.status, latencyMs };
}

export interface ClientTelemetry {
  roundTripLatencyMs: number;
  serverCalculationMs?: number;
  snapshotTimestamp?: string;
  modelVersion?: string;
}

/** Exchange a Supabase access token for a short-lived HttpOnly API cookie. */
export async function connectApiSession(accessToken: string, signal?: AbortSignal): Promise<void> {
  const response = await fetch(`${API_URL}/api/v1/session`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${accessToken}`,
      "X-MoatCheck-Request": "1",
    },
    credentials: "include",
    cache: "no-store",
    signal: signal ? AbortSignal.any([signal, AbortSignal.timeout(30_000)]) : AbortSignal.timeout(30_000),
  });
  if (!response.ok) {
    const body = await response.text();
    throw new Error(`API ${response.status}: ${body}`);
  }
}

/** Revoke the local browser session by expiring its HttpOnly API cookie. */
export async function clearApiSession(): Promise<void> {
  const response = await fetch(`${API_URL}/api/v1/session`, {
    method: "DELETE",
    headers: { "X-MoatCheck-Request": "1" },
    credentials: "include",
    cache: "no-store",
    signal: AbortSignal.timeout(30_000),
  });
  if (!response.ok) throw new Error(`API ${response.status}: ${await response.text()}`);
}

/** Read the current cookie-backed identity without retrieving an auth token. */
export async function getApiSession(): Promise<{ user_id: string; email: string | null } | null> {
  const response = await fetch(`${API_URL}/api/v1/session`, { credentials: "include", cache: "no-store" });
  if (response.status === 401) return null;
  if (!response.ok) throw new Error(`API ${response.status}: ${await response.text()}`);
  return response.json() as Promise<{ user_id: string; email: string | null }>;
}

/** Delete all persisted portfolios, backtests, stress runs, and recommendations for the signed-in user. */
export async function deleteMyPersistedData(): Promise<{ deleted: Record<string, number> }> {
  return request<{ deleted: Record<string, number> }>("/v1/privacy/data", { method: "DELETE" });
}

function publishTelemetry(value: ClientTelemetry): void {
  if (typeof window !== "undefined") {
    window.dispatchEvent(new CustomEvent<ClientTelemetry>("moatcheck:telemetry", { detail: value }));
  }
}

async function request<T>(path: string, init?: RequestInit, timeoutMs = 30_000): Promise<T> {
  const method = (init?.method ?? "GET").toUpperCase();
  const canRetry = method === "GET";

  let lastError: unknown;
  const startedAt = performance.now();
  for (let attempt = 1; attempt <= (canRetry ? MAX_ATTEMPTS : 1); attempt++) {
    try {
      const res = await fetch(`${API_URL}/api${path}`, {
        ...init,
        signal: init?.signal ? AbortSignal.any([init.signal, AbortSignal.timeout(timeoutMs)]) : AbortSignal.timeout(timeoutMs),
        headers: {
          "Content-Type": "application/json",
          "X-MoatCheck-Request": "1",
          ...init?.headers,
        },
        credentials: "include",
        cache: "no-store",
      });
      if (!res.ok) {
        if (canRetry && RETRYABLE_STATUS.has(res.status) && attempt < MAX_ATTEMPTS) {
          const retryAfter = Number(res.headers.get("retry-after"));
          const delay = Number.isFinite(retryAfter) && retryAfter > 0
            ? Math.min(retryAfter * 1000, 10_000)
            : RETRY_DELAY_MS;
          await sleep(delay);
          continue;
        }
        const body = await res.text();
        throw new Error(`API ${res.status}: ${body}`);
      }
      const payload = (await res.json()) as T;
      if (payload && typeof payload === "object" && "meta" in payload) {
        const meta = (payload as { meta?: ApiResponseMeta }).meta;
        if (meta) {
          const data = "data" in payload ? (payload as { data?: unknown }).data : undefined;
          const dataObject = data && typeof data === "object" ? (data as Record<string, unknown>) : undefined;
          publishTelemetry({
            roundTripLatencyMs: performance.now() - startedAt,
            serverCalculationMs: meta.execution_time_ms,
            snapshotTimestamp: meta.snapshot_timestamp ?? (typeof dataObject?.snapshot_timestamp === "string" ? dataObject.snapshot_timestamp : undefined),
            modelVersion: typeof dataObject?.model_version === "string" ? dataObject.model_version : undefined,
          });
        }
      }
      return payload;
    } catch (e) {
      lastError = e;
      if (init?.signal?.aborted || (e instanceof Error && e.name === "AbortError")) throw e;
      if (e instanceof Error && e.name === "TimeoutError") {
        throw new Error(path.includes("/proposals")
          ? "Portfolio discovery is taking longer than 90 seconds. Please retry; the analysis is limited to the liquid core universe."
          : "The API request timed out. Please retry.");
      }
      // A thrown non-ok response (above) is rethrown immediately, not retried
      // again here; this catch is for network-level failures (fetch itself
      // rejecting, e.g. the connection being refused while the host wakes up).
      if (e instanceof Error && e.message.startsWith("API ")) throw e;
      if (canRetry && attempt < MAX_ATTEMPTS) {
        await sleep(RETRY_DELAY_MS);
        continue;
      }
      throw e;
    }
  }
  throw lastError;
}

// `request()` throws Error("API {status}: {raw JSON body}"); pull the
// backend's `detail` message back out so callers can show its exact
// wording, not the wrapped "API 422: {...}" string.
export function backendErrorMessage(e: unknown): string {
  const raw = String(e);
  const jsonStart = raw.indexOf("{");
  if (jsonStart === -1) return raw;
  try {
    const parsed = JSON.parse(raw.slice(jsonStart));
    if (typeof parsed.detail === "string") return parsed.detail;
    if (parsed.detail && typeof parsed.detail.message === "string") return parsed.detail.message;
    if (Array.isArray(parsed.detail)) return parsed.detail.map((item: { msg?: string }) => item.msg ?? "Invalid input").join("; ");
  } catch {
    // fall through to raw message
  }
  return raw;
}

// V2 qualitative layer — a COUNT of recent classified events, never a score.
export interface QualitativeTally {
  positive: number;
  negative: number;
  neutral: number;
  window_days: number;
}

export type QualitativeCategory =
  | "dated_contract"
  | "m_and_a"
  | "regulatory_admission"
  | "guidance"
  | "backlog"
  | "governance_risk"
  | "activist_pressure"
  | "customer_concentration";
export type QualitativeSentiment = "positive" | "negative" | "neutral";
export type QualitativeSeverity = "low" | "medium" | "high";
export type QualitativeConfidence = "high" | "medium" | "low";
export type QualitativeSourceType = "edgar" | "ir_rss" | "newsletter" | "press";

export interface QualitativeEvent {
  id: number;
  ticker: string;
  category: QualitativeCategory;
  sentiment: QualitativeSentiment;
  severity: QualitativeSeverity;
  confidence: QualitativeConfidence;
  event_date: string | null;
  source_type: QualitativeSourceType;
  source_url: string | null;
  summary: string | null;
  note?: string | null;
  created_at: string;
}

export interface FeedStatusRow {
  id: number;
  ticker: string;
  source_type: QualitativeSourceType;
  feed_url: string | null;
  status: "ok" | "failed" | "stale";
  last_success_at: string | null;
  last_error: string | null;
  updated_at: string;
}

export interface Stock {
  ticker: string;
  name: string | null;
  sector: string | null;
  industry: string | null;
  currency: string | null;
  status: "active" | "pending_refresh";
  asset_type: "equity" | "etf";
  sector_benchmark_ticker: string | null;
  ir_rss_url: string | null;
  composite_score: number | null;
  computed_at?: string | null;
  updated_at: string | null;
  qualitative_tally?: QualitativeTally | null;
}

export interface ScreenerRow extends Stock {
  fundamental_score: number | null;
  risk_score: number | null;
  revenue_growth_yoy: number | null;
  market_cap: number | null;
  pe_trailing: number | null;
  debt_to_ebitda: number | null;
  qualitative_tally?: QualitativeTally;
}

export interface ScreenerResult {
  rows: ScreenerRow[];
  next_cursor: string | null;
  has_more: boolean;
  page_size: number;
  execution_time_ms?: number;
}

export interface ScoreBreakdown {
  fundamental?: { missing: string[]; components: Record<string, { weight: number; normalized: number }> };
  risk?: { missing: string[]; components: Record<string, number> };
  composite?: { missing: string[]; weights: Record<string, number> };
}

export interface StockDetail {
  stock: Stock;
  fundamentals: Record<string, number | string | null> | null;
  score: {
    volatility_annualized: number | null;
    sharpe_ratio: number | null;
    max_drawdown: number | null;
    fundamental_score: number | null;
    risk_score: number | null;
    composite_score: number | null;
    score_breakdown: ScoreBreakdown | null;
    computed_at: string;
  } | null;
  recent_prices: { date: string; close: number; volume: number | null }[];
}

export interface PricePoint {
  date: string;
  close: number;
}

export interface CurvePoint {
  date: string;
  value: number;
}

export interface AddStockResult {
  ticker: string;
  status: string;
  queued: boolean;
  priority: string;
}

export interface BacktestResult {
  error?: string;
  start_date?: string;
  top_n?: number | null;
  exact_tickers?: boolean;
  selected_tickers?: string[];
  scores_at_start?: Record<string, number>;
  tickers_excluded?: { ticker: string; reason: string }[];
  tickers_excluded_count?: number;
  tickers_scorable_count?: number;
  total_universe_count?: number;
  basket_coverage_ratio?: number | null;
  universe_scorable_ratio?: number | null;
  low_sample_warning?: boolean;
  low_sample_warning_message?: string | null;
  note?: string | null;
  basket?: { total_return: number | null; sharpe: number | null };
  benchmark?: { ticker: string; total_return: number | null; sharpe: number | null };
  benchmark_data_unavailable?: boolean;
  per_stock_vs_benchmarks?: {
    ticker: string;
    stock_return: number | null;
    benchmark_ticker: string;
    benchmark_return: number | null;
    vs_benchmark: number | null;
    sector_benchmark_ticker: string | null;
    sector_benchmark_available: boolean;
    sector_return: number | null;
    vs_sector_benchmark: number | null;
  }[];
  basket_curve?: CurvePoint[];
  benchmark_curve?: CurvePoint[];
}

export interface CompareResult {
  start_date: string;
  series: { ticker: string; total_return: number | null; curve: CurvePoint[] }[];
  excluded: { ticker: string; reason: string }[];
}

export interface CorrelationResult {
  tickers: string[];
  matrix: number[][];
}

export interface PortfolioWeightsResult {
  weights: Record<string, number>;
  excluded: { ticker: string; reason: string }[];
}

export interface EfficientFrontierPoint {
  target_return: number;
  volatility: number;
  weights: Record<string, number>;
}

export interface MaxSharpePoint {
  expected_return: number;
  volatility: number;
  sharpe_ratio: number | null;
  weights: Record<string, number>;
}

export interface EfficientFrontierResult {
  tickers: string[];
  excluded: { ticker: string; reason: string }[];
  frontier: EfficientFrontierPoint[];
  skipped_target_returns: number[];
  max_sharpe_point: MaxSharpePoint;
  note: string | null;
}

export interface BenchmarkOption {
  ticker: string;
  name: string | null;
  is_benchmark: boolean;
  sector_name?: string | null;
}

export const api = {
  health: requestHealth,
  listPortfolios: () => request<ApiEnvelope<SavedPortfolio[]>>("/v1/portfolios"),
  createPortfolioConfig: (body: Pick<SavedPortfolio, "name" | "tickers" | "target_weights"> & Partial<Pick<SavedPortfolio, "risk_constraints" | "optimization_parameters" | "description">>) =>
    request<ApiEnvelope<SavedPortfolio>>("/v1/portfolio/configs", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  updatePortfolioConfig: (portfolioId: string, body: Pick<SavedPortfolio, "name" | "tickers" | "target_weights"> & Partial<Pick<SavedPortfolio, "risk_constraints" | "optimization_parameters" | "description">>) =>
    request<ApiEnvelope<SavedPortfolio>>(`/v1/portfolio/configs/${encodeURIComponent(portfolioId)}`, {
      method: "PUT",
      body: JSON.stringify(body),
    }),
  deletePortfolioConfig: (portfolioId: string) =>
    request<ApiEnvelope<{ deleted: boolean; portfolio_id: string }>>(`/v1/portfolio/configs/${encodeURIComponent(portfolioId)}`, {
      method: "DELETE",
    }),
  previewAdvisor: (body: AdvisorPreviewRequest, signal?: AbortSignal) =>
    request<ApiEnvelope<AdvisorRecommendationResult>>("/v1/advisor/recommendations/preview", {
      method: "POST",
      body: JSON.stringify(body),
      signal,
    }),
  getPortfolioTracking: (portfolioId: string, signal?: AbortSignal) =>
    request<ApiEnvelope<PortfolioTrackingData>>(`/v1/advisor/portfolios/${encodeURIComponent(portfolioId)}/tracking`, { signal }),
  savePortfolioHoldings: (portfolioId: string, holdings: PortfolioHoldingInput[], signal?: AbortSignal) =>
    request<ApiEnvelope<{ holdings: PortfolioHoldingInput[] }>>(`/v1/advisor/portfolios/${encodeURIComponent(portfolioId)}/holdings`, {
      method: "PUT", body: JSON.stringify({ holdings }), signal,
    }),
  savePortfolioTrackingSettings: (portfolioId: string, body: Pick<PortfolioTrackingData, "monthly_dca_amount" | "target_horizon_years">, signal?: AbortSignal) =>
    request<ApiEnvelope<Pick<PortfolioTrackingData, "monthly_dca_amount" | "target_horizon_years">>>(`/v1/advisor/portfolios/${encodeURIComponent(portfolioId)}/tracking-settings`, {
      method: "PUT", body: JSON.stringify(body), signal,
    }),
  createAdvisorRecommendation: (portfolioId: string, body: { market: AdvisorMarketSnapshot; policy?: Partial<AdvisorPolicy>; email_notification?: boolean }, signal?: AbortSignal) =>
    request<ApiEnvelope<PersistedAdvisorRecommendation>>(`/v1/advisor/portfolios/${encodeURIComponent(portfolioId)}/recommendations`, {
      method: "POST",
      body: JSON.stringify(body),
      signal,
    }),
  latestAdvisorRecommendation: (portfolioId: string, signal?: AbortSignal) =>
    request<ApiEnvelope<PersistedAdvisorRecommendation>>(`/v1/advisor/portfolios/${encodeURIComponent(portfolioId)}/recommendations/latest`, { signal }),
  discoverPortfolioProposals: (portfolioId: string) =>
    request<ApiEnvelope<PortfolioDiscoveryData>>(`/v1/advisor/portfolios/${encodeURIComponent(portfolioId)}/proposals`, undefined, 90_000),
  applyPortfolioProposal: (portfolioId: string, proposalId: string, targetWeights: Record<string, number>) => {
    const tickers = Object.keys(targetWeights);
    return request<ApiEnvelope<SavedPortfolio & { trade_execution: string; price_refreshes?: { ticker: string; status: string }[] }>>(
      `/v1/advisor/portfolios/${encodeURIComponent(portfolioId)}/apply-proposal`,
      { method: "POST", body: JSON.stringify({ proposal_id: proposalId, tickers, target_weights: targetWeights }) },
    );
  },
  acknowledgeAdvisorRecommendation: (recommendationId: string) =>
    request<ApiEnvelope<PersistedAdvisorRecommendation>>(`/v1/advisor/recommendations/${encodeURIComponent(recommendationId)}/acknowledge`, {
      method: "POST",
      body: JSON.stringify({}),
    }),
  optimizeFrontier: (body: QuantPortfolioRequest, signal?: AbortSignal) =>
    request<ApiEnvelope<{ frontier: QuantFrontierPoint[] }>>("/v1/portfolio/optimize", {
      method: "POST",
      body: JSON.stringify(body),
      signal,
    }),
  optimizeMaxSharpe: (body: QuantPortfolioRequest, signal?: AbortSignal) =>
    request<ApiEnvelope<PortfolioOptimizationData>>("/v1/portfolio/markowitz/max-sharpe", {
      method: "POST",
      body: JSON.stringify(body),
      signal,
    }),
  optimizeMinVariance: (body: QuantPortfolioRequest, signal?: AbortSignal) =>
    request<ApiEnvelope<PortfolioOptimizationData>>("/v1/portfolio/markowitz/min-variance", {
      method: "POST",
      body: JSON.stringify(body),
      signal,
    }),
  riskParity: (body: { tickers: string[]; covariance_matrix: number[][]; target_risk_budget?: number[] }) =>
    request<ApiEnvelope<Record<string, unknown>>>("/v1/portfolio/risk-parity", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  portfolioRisk: (body: PortfolioRiskRequest, signal?: AbortSignal) =>
    request<ApiEnvelope<PortfolioRiskData>>("/v1/risk/evaluate", { method: "POST", body: JSON.stringify(body), signal }),
  liquidityAdjustedVar: (body: LiquidityVaRRequest, signal?: AbortSignal) =>
    request<ApiEnvelope<LiquidityVaRData>>("/v1/risk/liquidity-adjusted-var", {
      method: "POST",
      body: JSON.stringify(body),
      signal,
    }),
  historicalStress: (body: HistoricalStressRequest) =>
    request<ApiEnvelope<HistoricalStressData>>("/v1/risk/stress/historical", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  historicalScenarioCatalog: (signal?: AbortSignal) =>
    request<ApiEnvelope<{ scenarios: CrisisScenarioCatalogEntry[] }>>("/v1/risk/stress/scenarios", { signal }),
  replayHistoricalCrisis: (scenarioId: string, weights?: number[], signal?: AbortSignal) =>
    request<ApiEnvelope<HistoricalCrisisReplayData>>(
      `/v1/risk/stress/scenarios/${encodeURIComponent(scenarioId)}/replay`,
      { method: "POST", body: JSON.stringify(weights ? { weights } : {}), signal },
    ),
  rebalanceSimulation: (body: RebalanceSimulationRequest, signal?: AbortSignal) =>
    request<ApiEnvelope<RebalanceSimulationData>>("/v1/backtest/run", {
      method: "POST",
      body: JSON.stringify(body),
      signal,
    }),
  executionCosts: (body: ExecutionCostRequest, signal?: AbortSignal) =>
    request<ApiEnvelope<ExecutionCostData>>("/v1/backtest/execution-costs", {
      method: "POST",
      body: JSON.stringify(body),
      signal,
    }),
  listStocks: () => request<Stock[]>("/stocks"),
  listBenchmarks: () => request<BenchmarkOption[]>("/stocks/benchmarks"),
  stockDetails: (tickers: string[], signal?: AbortSignal) => request<StockDetail[]>(`/stocks/details?tickers=${encodeURIComponent(tickers.join(","))}`, { signal }),
  addStock: (ticker: string, adminKey: string, assetType: "equity" | "etf" = "equity") =>
    request<AddStockResult>("/stocks", {
      method: "POST",
      headers: { "X-Admin-Key": adminKey },
      body: JSON.stringify({ ticker, asset_type: assetType }),
    }),
  updateStock: (
    ticker: string,
    adminKey: string,
    body: { sector_benchmark_ticker?: string | null; ir_rss_url?: string | null }
  ) =>
    request<Stock>(`/stocks/${ticker}`, {
      method: "PATCH",
      headers: { "X-Admin-Key": adminKey },
      body: JSON.stringify(body),
    }),
  // V2 qualitative layer
  stockQualitative: (ticker: string, params?: { category?: string; min_confidence?: string }) => {
    const q = new URLSearchParams();
    if (params?.category) q.set("category", params.category);
    if (params?.min_confidence) q.set("min_confidence", params.min_confidence);
    const qs = q.toString();
    return request<{ ticker: string; window_days: number; events: QualitativeEvent[] }>(
      `/stocks/${ticker}/qualitative${qs ? `?${qs}` : ""}`
    );
  },
  qualitativeTally: (ticker: string) =>
    request<QualitativeTally>(`/qualitative/tally?ticker=${encodeURIComponent(ticker)}`),
  qualitativeFeed: (params?: { category?: string; ticker?: string; limit?: number }) => {
    const q = new URLSearchParams();
    if (params?.category) q.set("category", params.category);
    if (params?.ticker) q.set("ticker", params.ticker);
    if (params?.limit !== undefined) q.set("limit", String(params.limit));
    const qs = q.toString();
    return request<{ events: QualitativeEvent[]; window_days: number }>(
      `/qualitative${qs ? `?${qs}` : ""}`
    );
  },
  feedStatus: (adminKey: string, onlyProblems = false) =>
    request<{ feeds: FeedStatusRow[] }>(`/feed-status${onlyProblems ? "?only_problems=true" : ""}`, {
      headers: { "X-Admin-Key": adminKey },
    }),
  qualitativeRefresh: (adminKey: string) =>
    request<Record<string, unknown>>("/qualitative/refresh", {
      method: "POST",
      headers: { "X-Admin-Key": adminKey },
    }),
  stockDetail: (ticker: string, signal?: AbortSignal) => request<StockDetail>(`/stocks/${ticker}`, { signal }),
  priceHistories: (tickers: string[], signal?: AbortSignal) => request<Record<string, PricePoint[]>>(`/stocks/history?tickers=${encodeURIComponent(tickers.join(","))}`, { signal }),
  priceHistory: (ticker: string, signal?: AbortSignal) => request<PricePoint[]>(`/stocks/${ticker}/history`, { signal }),
  screener: (params: {
    sector?: string;
    min_score?: number;
    min_growth?: number;
    min_risk_score?: number;
    market_cap_min?: number;
    market_cap_max?: number;
    pe_max?: number;
    debt_to_ebitda_max?: number;
    include_etfs?: boolean;
    sort_by?: keyof Pick<ScreenerRow, "composite_score" | "fundamental_score" | "risk_score" | "revenue_growth_yoy" | "market_cap" | "pe_trailing" | "debt_to_ebitda">;
    sort_dir?: "asc" | "desc";
    limit?: number;
    cursor?: string;
  }, signal?: AbortSignal) => {
    const q = new URLSearchParams();
    if (params.sector) q.set("sector", params.sector);
    if (params.min_score !== undefined) q.set("min_score", String(params.min_score));
    if (params.min_growth !== undefined) q.set("min_growth", String(params.min_growth));
    if (params.min_risk_score !== undefined) q.set("min_risk_score", String(params.min_risk_score));
    if (params.market_cap_min !== undefined) q.set("market_cap_min", String(params.market_cap_min));
    if (params.market_cap_max !== undefined) q.set("market_cap_max", String(params.market_cap_max));
    if (params.pe_max !== undefined) q.set("pe_max", String(params.pe_max));
    if (params.debt_to_ebitda_max !== undefined) q.set("debt_to_ebitda_max", String(params.debt_to_ebitda_max));
    if (params.include_etfs) q.set("include_etfs", "true");
    if (params.sort_by) q.set("sort_by", params.sort_by);
    if (params.sort_dir) q.set("sort_dir", params.sort_dir);
    if (params.limit !== undefined) q.set("limit", String(params.limit));
    if (params.cursor) q.set("cursor", params.cursor);
    const qs = q.toString();
    return request<ScreenerResult>(`/v1/screener${qs ? `?${qs}` : ""}`, { signal });
  },
  ingestTicker: (ticker: string) => request<{ ticker: string; name: string | null; currency: string | null; asset_type: "equity" | "etf"; status: string; liquidity_verified: boolean; adv_usd: number | null }>("/v1/universe/ingest", {
    method: "POST",
    body: JSON.stringify({ ticker }),
  }),
  searchTicker: (ticker: string, signal?: AbortSignal) => request<{ found: boolean; ticker: string; name: string | null; currency?: string | null; asset_type?: "equity" | "etf" }>(`/v1/universe/search?query=${encodeURIComponent(ticker)}`, { signal }),
  refresh: (tickers: string[]) =>
    request<{ results: { ticker: string; ok: boolean; error?: string }[] }>("/refresh", {
      method: "POST",
      body: JSON.stringify({ tickers }),
    }),
  recompute: () => request<{ results: unknown[] }>("/score/recompute", { method: "POST" }),
  backtest: (body: { start_date: string; top_n?: number; benchmark: string; tickers?: string[] }) =>
    request<BacktestResult>("/backtest", { method: "POST", body: JSON.stringify(body) }),
  compare: (tickers: string[], startDate: string) => {
    const q = new URLSearchParams({ tickers: tickers.join(","), start_date: startDate });
    return request<CompareResult>(`/compare?${q.toString()}`);
  },
  correlation: (tickers: string[], signal?: AbortSignal) => {
    const q = new URLSearchParams({ tickers: tickers.join(",") });
    return request<CorrelationResult>(`/correlation?${q.toString()}`, { signal });
  },
  portfolioWeights: (tickers: string[]) => {
    const q = new URLSearchParams({ tickers: tickers.join(",") });
    return request<PortfolioWeightsResult>(`/portfolio/weights?${q.toString()}`);
  },
  efficientFrontier: (tickers: string[]) => {
    const q = new URLSearchParams({ tickers: tickers.join(",") });
    return request<EfficientFrontierResult>(`/portfolio/efficient-frontier?${q.toString()}`);
  },
};
