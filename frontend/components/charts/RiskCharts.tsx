"use client";

import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export interface DistributionPoint {
  range: string;
  count: number;
  tail: boolean;
}

export interface StressPathPoint {
  date: string;
  drawdown: number;
}

export interface VolatilityPathPoint {
  date: string;
  portfolio: number | null;
}

export function ReturnDistributionChart({ data }: { data: DistributionPoint[] }) {
  return (
    <ResponsiveContainer>
      <BarChart data={data} margin={{ top: 10, right: 12, left: -20, bottom: 2 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#64748b" opacity={0.2} />
        <XAxis dataKey="range" tick={{ fontSize: 9 }} interval="preserveStartEnd" />
        <YAxis allowDecimals={false} tick={{ fontSize: 10 }} />
        <Tooltip />
        <Bar dataKey="count" name="Sessions" radius={[10, 10, 0, 0]}>
          {data.map((item) => <Cell key={item.range} fill={item.tail ? "#FF5252" : "#22C55E"} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

export function CrisisDrawdownChart({ data }: { data: StressPathPoint[] }) {
  return (
    <ResponsiveContainer>
      <AreaChart data={data} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
        <defs><linearGradient id="crisisDrawdown" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#FF5252" stopOpacity={0.24} /><stop offset="100%" stopColor="#FF5252" stopOpacity={0.01} /></linearGradient></defs>
        <CartesianGrid strokeDasharray="3 3" stroke="#64748b" opacity={0.2} />
        <XAxis dataKey="date" tick={{ fontSize: 10 }} minTickGap={20} />
        <YAxis tickFormatter={(value: number) => `${value}%`} tick={{ fontSize: 10 }} />
        <Tooltip formatter={(value: number) => `${Number(value).toFixed(2)}%`} />
        <ReferenceLine y={0} stroke="#94a3b8" />
        <Area type="monotone" dataKey="drawdown" name="Drawdown" stroke="#FF5252" fill="url(#crisisDrawdown)" strokeWidth={2} dot={false} />
      </AreaChart>
    </ResponsiveContainer>
  );
}

export function HistoricalVolatilityChart({ data }: { data: VolatilityPathPoint[] }) {
  return (
    <ResponsiveContainer>
      <LineChart data={data} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#64748b" opacity={0.2} />
        <XAxis dataKey="date" tick={{ fontSize: 10 }} minTickGap={20} />
        <YAxis tickFormatter={(value: number) => `${value}%`} tick={{ fontSize: 10 }} />
        <Tooltip formatter={(value: unknown) => value == null ? "Not enough data" : `${Number(value).toFixed(1)}%`} />
        <Line connectNulls type="monotone" dataKey="portfolio" name="Benchmark volatility" stroke="#38bdf8" strokeWidth={2} dot={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}
