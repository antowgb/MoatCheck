"use client";

import { useEffect, useState } from "react";
import { Save } from "lucide-react";
import Card from "@/components/Card";
import type { PortfolioHoldingInput } from "@/lib/api";

type HoldingDraft = { quantity: string; pru: string; currency: string; acquired_at: string };

export default function PortfolioHoldingsEditor({
  tickers, holdings, currencies, busy, onSave,
}: {
  tickers: string[]; holdings: PortfolioHoldingInput[]; currencies: Record<string, string>; busy?: boolean;
  onSave: (holdings: PortfolioHoldingInput[]) => void;
}) {
  const [values, setValues] = useState<Record<string, HoldingDraft>>({});
  const [currencyError, setCurrencyError] = useState<string | null>(null);
  const [confirmedQuotePrices, setConfirmedQuotePrices] = useState(false);
  const currencyMismatches = tickers.flatMap((ticker) => {
    const saved = holdings.find((holding) => holding.ticker === ticker);
    const quoteCurrency = currencies[ticker]?.trim().toUpperCase();
    return saved && quoteCurrency && saved.currency.toUpperCase() !== quoteCurrency
      ? [{ ticker, savedCurrency: saved.currency.toUpperCase(), quoteCurrency }]
      : [];
  });
  useEffect(() => {
    const initial = Object.fromEntries(tickers.map((ticker) => {
      const item = holdings.find((holding) => holding.ticker === ticker);
      return [ticker, {
        quantity: item ? String(item.quantity) : "",
        pru: item ? String(item.average_buy_price) : "",
        currency: (currencies[ticker] || item?.currency || "USD").trim().toUpperCase(),
        acquired_at: item?.acquired_at ?? new Date().toISOString().slice(0, 10),
      }];
    }));
    setValues(initial);
    setConfirmedQuotePrices(false);
  }, [tickers.join("|"), holdings, currencies]);

  function save() {
    const nextHoldings = tickers.flatMap((ticker) => {
      const row = values[ticker];
      const quantity = Number(row?.quantity), average_buy_price = Number(row?.pru);
      if (!Number.isFinite(quantity) || quantity <= 0 || !Number.isFinite(average_buy_price) || average_buy_price <= 0) return [];
      return [{ ticker, quantity, average_buy_price, currency: row.currency, acquired_at: row.acquired_at }];
    });
    if (nextHoldings.some((holding) => !/^[A-Z]{3}$/.test(holding.currency))) {
      setCurrencyError("Choose the purchase currency for each holding before saving.");
      return;
    }
    if (currencyMismatches.length && !confirmedQuotePrices) {
      setCurrencyError("Confirm that each average buy price has been converted to the asset's quote currency before saving.");
      return;
    }
    setCurrencyError(null);
    onSave(nextHoldings);
  }

  return <Card title="Track what you own" className="col-span-12 xl:col-span-8">
    <p className="mb-3 text-xs leading-relaxed text-slate-500">Enter your shares or units, average buy price, and when you bought them. Leave quantity and price blank for an asset you do not own. Acquisition dates are used to enforce the minimum holding period on trade signals.</p>
    <div className="mb-4 grid gap-2 rounded-xl border border-cyan-300/10 bg-cyan-300/[0.035] p-3 text-[11px] leading-relaxed text-slate-400 sm:grid-cols-2">
      <p><span className="font-semibold text-cyan-200">Quantity</span> means the number of shares or units, not the amount invested. For example, enter 3.5 if you own three and a half shares.</p>
      <p><span className="font-semibold text-cyan-200">Your entries stay fixed</span> until you change them. Market value and unrealized P&amp;L use the latest stored close when this page loads or you request a preview. These are not live prices.</p>
    </div>
    {currencyMismatches.length > 0 && <div className="mb-4 rounded-xl border border-amber-300/20 bg-amber-300/[0.05] p-3 text-[11px] leading-relaxed text-amber-100"><p>Saved currencies differ from the assets' quote currencies: {currencyMismatches.map(({ ticker, savedCurrency, quoteCurrency }) => `${ticker} ${savedCurrency} → ${quoteCurrency}`).join(", ")}.</p><p className="mt-1 text-amber-100/70">The currency is now locked to each quote currency. Convert each average buy price to that currency before saving; MoatCheck does not have historical FX rates to convert it for you.</p><label className="mt-2 flex items-start gap-2"><input type="checkbox" checked={confirmedQuotePrices} onChange={(event) => { setConfirmedQuotePrices(event.target.checked); setCurrencyError(null); }} className="mt-0.5 accent-amber-300" /><span>I confirm the average buy prices below use their displayed quote currencies.</span></label></div>}
    <div className="overflow-x-auto"><table className="w-full min-w-[800px] text-left text-xs">
      <thead><tr className="border-b border-white/[0.07] text-[10px] uppercase tracking-wider text-slate-500"><th className="pb-2">Asset</th><th className="pb-2 text-right">Quantity</th><th className="pb-2 text-right">Average buy price</th><th className="pb-2 text-right">Bought on</th><th className="pb-2 text-right">Quote currency</th></tr></thead>
      <tbody>{tickers.map((ticker) => <tr key={ticker} className="border-b border-white/[0.045] last:border-0">
        <th className="py-2.5 font-medium text-slate-200">{ticker}</th>
        <td className="py-2 text-right"><input aria-label={`${ticker} quantity`} type="number" min="0" step="any" value={values[ticker]?.quantity ?? ""} onChange={(event) => setValues((previous) => ({ ...previous, [ticker]: { ...previous[ticker], quantity: event.target.value } }))} placeholder="0" className="w-28 rounded-lg border border-white/[0.08] bg-black/20 px-2.5 py-2 text-right font-mono text-slate-100 outline-none focus:border-cyan-300/30" /></td>
        <td className="py-2 text-right"><input aria-label={`${ticker} average buy price`} type="number" min="0" step="any" value={values[ticker]?.pru ?? ""} onChange={(event) => setValues((previous) => ({ ...previous, [ticker]: { ...previous[ticker], pru: event.target.value } }))} placeholder="0.00" className="w-28 rounded-lg border border-white/[0.08] bg-black/20 px-2.5 py-2 text-right font-mono text-slate-100 outline-none focus:border-cyan-300/30" /></td>
        <td className="py-2 text-right"><input aria-label={`${ticker} acquisition date`} type="date" max={new Date().toISOString().slice(0, 10)} value={values[ticker]?.acquired_at ?? ""} onChange={(event) => setValues((previous) => ({ ...previous, [ticker]: { ...previous[ticker], acquired_at: event.target.value } }))} className="w-36 rounded-lg border border-white/[0.08] bg-[#111720] px-2 py-2 text-right font-mono text-slate-200 outline-none focus:border-cyan-300/30" /></td>
        <td className="py-2 text-right"><span aria-label={`${ticker} quote currency, locked`} className="inline-flex min-w-16 justify-center rounded-lg border border-white/[0.08] bg-[#111720] px-2.5 py-2 font-mono text-slate-200">{values[ticker]?.currency || currencies[ticker] || "USD"}</span></td>
      </tr>)}</tbody>
    </table></div>
    {currencyError && <p role="alert" className="mt-3 text-xs text-rose-300">{currencyError}</p>}
    <button type="button" disabled={busy} onClick={save} className="mt-4 inline-flex items-center gap-1.5 rounded-lg border border-emerald-300/20 bg-emerald-300/[0.06] px-3 py-2 text-[11px] font-semibold text-emerald-200 hover:bg-emerald-300/[0.12] disabled:opacity-50"><Save size={13} />Save holdings</button>
  </Card>;
}
