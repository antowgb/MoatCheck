/** Pure arithmetic contracts shared by portfolio screens. Returns are fractions. */
export function inferPeriodsPerYear(dates: string[]): number {
  const gaps = dates.slice(1).map((date, index) =>
    (Date.parse(date) - Date.parse(dates[index])) / 86_400_000,
  );
  if (!gaps.length || gaps.some((gap) => !Number.isFinite(gap) || gap <= 0)) {
    throw new Error("Price dates must be valid and strictly increasing.");
  }
  const sorted = [...gaps].sort((a, b) => a - b);
  const median = sorted[Math.floor(sorted.length / 2)];
  // Weekends do not turn exchange-traded daily observations into 365 sessions.
  return median <= 4 ? 252 : Math.max(1, Math.round(365.25 / median));
}

/** Arithmetic sample moments for Markowitz and weighted simple-return risk. */
export function portfolioReturnInputs(prices: number[][], periodsPerYear: number) {
  const assets = prices[0]?.length ?? 0;
  if (prices.length < 3 || !assets || !Number.isFinite(periodsPerYear) || periodsPerYear <= 0
    || prices.some((row) => row.length !== assets || row.some((price) => !Number.isFinite(price) || price <= 0))) {
    throw new Error("At least three complete, positive price observations are required.");
  }
  const logReturns = prices.slice(1).map((row, t) => row.map((price, i) => Math.log(price) - Math.log(prices[t][i])));
  const returns = logReturns.map((row) => row.map(Math.expm1));
  if (returns.some((row) => row.some((value) => !Number.isFinite(value)))) {
    throw new Error("Price changes exceed the supported numerical range.");
  }
  const means = prices[0].map((_, i) => returns.reduce((sum, row) => sum + row[i], 0) / returns.length);
  const expectedReturns = means.map((mean) => mean * periodsPerYear);
  const covariance = means.map((leftMean, i) => means.map((rightMean, j) =>
    returns.reduce((sum, row) => sum + (row[i] - leftMean) * (row[j] - rightMean), 0)
      / (returns.length - 1) * periodsPerYear,
  ));
  if (![...expectedReturns, ...covariance.flat()].every(Number.isFinite)) {
    throw new Error("Historical return moments exceed the supported numerical range.");
  }
  return { returns, logReturns, expectedReturns, covariance };
}

/** Drawdowns use the running peak, including initial capital, never a future peak. */
export function drawdownPath(wealth: number[], initialWealth: number): number[] {
  if (!Number.isFinite(initialWealth) || initialWealth <= 0
    || wealth.some((value) => !Number.isFinite(value) || value < 0)) {
    throw new Error("Wealth must be finite and non-negative, with positive initial capital.");
  }
  let peak = initialWealth;
  return wealth.map((value) => { peak = Math.max(peak, value); return value / peak - 1; });
}
