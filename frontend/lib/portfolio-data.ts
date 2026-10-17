import { api, type PricePoint } from "@/lib/api";
import { inferPeriodsPerYear, portfolioReturnInputs } from "@/lib/portfolio-math";

export interface AlignedPortfolioHistory {
  tickers: string[];
  dates: string[];
  prices: number[][];
  /** Simple returns, suitable for weighted portfolio aggregation. */
  returns: number[][];
  logReturns: number[][];
  expectedReturns: number[];
  covariance: number[][];
  periodsPerYear: number;
}

/** Align stored price histories to common dates and derive annualized inputs. */
export async function loadAlignedPortfolioHistory(tickers: string[], signal?: AbortSignal): Promise<AlignedPortfolioHistory> {
  if (tickers.length === 0) throw new Error("The selected portfolio has no assets.");

  const historyByTicker = await api.priceHistories(tickers, signal);
  const histories = tickers.map((ticker) => historyByTicker[ticker] ?? []);
  const priceMaps = histories.map((history: PricePoint[]) => new Map(
    history
      .filter((point) => Number.isFinite(point.close) && point.close > 0)
      .map((point) => [point.date.slice(0, 10), point.close] as const),
  ));
  const commonDates = [...priceMaps[0].keys()]
    .filter((date) => priceMaps.every((prices) => prices.has(date)))
    .sort();
  if (commonDates.length < 3) {
    throw new Error("There are not enough shared price observations for this portfolio. Check that every selected asset has stored price history.");
  }

  const dates = commonDates.slice(-756);
  const prices = dates.map((date) => priceMaps.map((asset) => asset.get(date) as number));
  const periodsPerYear = inferPeriodsPerYear(dates);
  const moments = portfolioReturnInputs(prices, periodsPerYear);
  return { tickers, dates, prices, periodsPerYear, ...moments };
}

/** Select approximately monthly observations from an aligned history panel. */
export function monthlyPricePanel(prices: number[][], periodsPerYear: number): number[][] {
  if (prices.length <= 2) return prices;
  const step = Math.max(1, Math.round(periodsPerYear / 12));
  const sampled = prices.filter((_, index) => index % step === 0);
  const last = prices[prices.length - 1];
  if (sampled[sampled.length - 1] !== last) sampled.push(last);
  return sampled;
}
