import {
  Activity, BarChart3, Briefcase, GitCompare, History, LayoutDashboard,
  ListFilter, Newspaper, ShieldAlert, WandSparkles, type LucideIcon,
} from "lucide-react";

export type AppNavigationItem = { href: string; label: string; icon: LucideIcon; group?: string };

export const APP_NAVIGATION: AppNavigationItem[] = [
  { href: "/", label: "Dashboard Overview", icon: LayoutDashboard },
  { href: "/optimizer", label: "Portfolio Optimizer", icon: Briefcase },
  { href: "/risk", label: "Risk & Stress Lab", icon: ShieldAlert },
  { href: "/backtest", label: "Backtest Tear Sheet", icon: History },
  { href: "/advisor", label: "Portfolio Advisor", icon: WandSparkles },
  { href: "/portfolio", label: "My Portfolio", icon: Briefcase, group: "Portfolio" },
  { href: "/screener", label: "Equity Screener", icon: ListFilter, group: "Research" },
  { href: "/compare", label: "Compare Assets", icon: GitCompare },
  { href: "/qualitative", label: "Qualitative Research", icon: Newspaper },
  { href: "/methodology", label: "Methodology", icon: BarChart3 },
  { href: "/admin", label: "Data Operations", icon: Activity },
];
