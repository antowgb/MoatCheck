"use client";

import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export interface BacktestChartPoint {
  period: string;
  pnl: number;
  wealth: number;
  drawdown: number;
}

export interface CostChartPoint {
  source: string;
  amount: number;
}

export function PortfolioValueChart({ data }: { data: BacktestChartPoint[] }) {
  return (
    <ResponsiveContainer>
      <AreaChart data={data} margin={{ top: 10, right: 12, left: 0, bottom: 0 }}>
        <defs><linearGradient id="pnlFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#22C55E" stopOpacity={0.34} /><stop offset="95%" stopColor="#22C55E" stopOpacity={0.01} /></linearGradient></defs>
        <CartesianGrid strokeDasharray="3 3" stroke="#64748b" opacity={0.2} />
        <XAxis dataKey="period" tick={{ fontSize: 10 }} />
        <YAxis tickFormatter={(value: number) => `€${(value / 1000).toFixed(0)}k`} tick={{ fontSize: 10 }} />
        <Tooltip formatter={(value: number) => `€${Number(value).toLocaleString("en-US", { maximumFractionDigits: 0 })}`} />
        <Area type="monotone" dataKey="pnl" name="Net PnL" stroke="#22C55E" fill="url(#pnlFill)" strokeWidth={2} />
      </AreaChart>
    </ResponsiveContainer>
  );
}

export function PortfolioDrawdownChart({ data }: { data: BacktestChartPoint[] }) {
  return (
    <ResponsiveContainer>
      <AreaChart data={data} margin={{ top: 10, right: 12, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#64748b" opacity={0.2} />
        <XAxis dataKey="period" tick={{ fontSize: 10 }} />
        <YAxis tickFormatter={(value: number) => `${(value * 100).toFixed(0)}%`} tick={{ fontSize: 10 }} />
        <Tooltip formatter={(value: number) => `${(Number(value) * 100).toFixed(2)}%`} />
        <Area type="monotone" dataKey="drawdown" name="Drawdown" stroke="#fb7185" fill="#fb7185" fillOpacity={0.15} />
      </AreaChart>
    </ResponsiveContainer>
  );
}

export function TradingCostsChart({ data }: { data: CostChartPoint[] }) {
  return (
    <ResponsiveContainer>
      <BarChart data={data} margin={{ top: 4, right: 10, left: -15, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#64748b" opacity={0.2} />
        <XAxis dataKey="source" tick={{ fontSize: 10 }} />
        <YAxis tickFormatter={(value: number) => `€${value}`} tick={{ fontSize: 10 }} />
        <Tooltip formatter={(value: number) => `€${Number(value).toFixed(2)}`} />
        <Bar dataKey="amount" name="Cost" radius={[10, 10, 0, 0]}>
          {data.map((item, index) => <Cell key={item.source} fill={["#fbbf24", "#FF5252", "#38bdf8", "#8B5CF6"][index % 4]} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
