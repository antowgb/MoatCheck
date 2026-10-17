import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import ts from "typescript";

const source = readFileSync(new URL("./portfolio-math.ts", import.meta.url), "utf8");
const { outputText } = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2020 } });
const { inferPeriodsPerYear, portfolioReturnInputs, drawdownPath } = await import(`data:text/javascript;base64,${Buffer.from(outputText).toString("base64")}`);
const close = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-10, `${actual} != ${expected}`);

test("exchange sessions annualize at 252 despite weekends", () => {
  assert.equal(inferPeriodsPerYear(["2026-10-08", "2026-10-09", "2026-10-12", "2026-10-13"]), 252);
});
test("weekly history annualizes at 52 observations", () => {
  assert.equal(inferPeriodsPerYear(["2026-10-02", "2026-10-09", "2026-10-16"]), 52);
});
test("invalid, duplicate and reversed dates fail clearly", () => {
  for (const dates of [[], ["2026-10-09"], ["invalid", "2026-10-10"], ["2026-10-09", "2026-10-09"], ["2026-10-10", "2026-10-09"]]) {
    assert.throws(() => inferPeriodsPerYear(dates), /strictly increasing/);
  }
});
test("weighted risk receives simple returns and Markowitz receives arithmetic moments", () => {
  const result = portfolioReturnInputs([[100, 100], [110, 90], [99, 99]], 252);
  close(result.returns[0][0], .1);
  close(result.returns[0][1], -.1);
  close(result.returns[0][0] * .5 + result.returns[0][1] * .5, 0);
  close(result.expectedReturns[0], 0);
  close(result.covariance[0][0], .02 * 252);
  close(result.logReturns[0][0], Math.log(1.1));
});
test("constant single-asset history produces finite zero variance", () => {
  const result = portfolioReturnInputs([[100], [100], [100]], 52);
  assert.deepEqual(result.expectedReturns, [0]);
  assert.deepEqual(result.covariance, [[0]]);
});
test("empty, sparse, ragged and invalid price buffers are rejected", () => {
  for (const prices of [[], [[1]], [[1], [2]], [[1], [2, 3], [4]], [[1], [0], [1]], [[1], [NaN], [1]], [[1], [Infinity], [1]]]) {
    assert.throws(() => portfolioReturnInputs(prices, 252));
  }
  assert.throws(() => portfolioReturnInputs([[1], [2], [3]], NaN));
});
test("unrepresentable simple returns fail instead of displaying Infinity", () => {
  assert.throws(() => portfolioReturnInputs([[1e-308], [1e308], [1e308]], 252), /numerical range/);
});
test("drawdown uses only peaks already reached and includes starting capital", () => {
  const values = drawdownPath([100, 90, 110, 99], 100);
  [0, -.1, 0, -.1].forEach((value, i) => close(values[i], value));
  close(Math.min(...values), -.1);
  assert.deepEqual(drawdownPath([0], 100), [-1]);
  assert.deepEqual(drawdownPath([], 100), []);
  assert.throws(() => drawdownPath([NaN], 100));
});
