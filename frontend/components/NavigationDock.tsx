"use client";

import { useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { Ellipsis, type LucideIcon } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { APP_NAVIGATION } from "@/components/app-navigation";

function activeRoute(pathname: string, href: string) {
  return pathname === href || (href !== "/" && pathname.startsWith(`${href}/`));
}

function NavItem({ href, label, icon: Icon, compact }: { href: string; label: string; icon: LucideIcon; compact?: boolean }) {
  const pathname = usePathname();
  const active = activeRoute(pathname, href);
  const shortLabel: Record<string, string> = {
    "/": "Home", "/optimizer": "Optimizer", "/risk": "Risk", "/backtest": "Backtest", "/advisor": "Advisor",
  };
  return <Link href={href} aria-label={label} aria-current={active ? "page" : undefined} title={label}
    onClick={() => document.querySelectorAll("nav[aria-label='Workspace navigation'] details[open]").forEach((menu) => menu.removeAttribute("open"))}
    className={`relative flex min-w-0 rounded-2xl transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400 ${compact ? "h-14 flex-1 flex-col items-center justify-center gap-1 px-1.5 py-2 text-[10px]" : "w-full flex-row items-center justify-start gap-3 px-3 py-2.5 text-xs"} ${active ? "text-emerald-200" : "text-slate-500 hover:bg-white/[0.04] hover:text-slate-100"}`}>
    {active && <motion.span layoutId="active-nav-dock" className="absolute inset-0 rounded-2xl border border-emerald-300/15 bg-emerald-400/[0.1]" transition={{ type: "spring", stiffness: 440, damping: 34 }} />}
    <Icon size={compact ? 22 : 15} strokeWidth={compact ? 1.8 : 2} className={`relative z-[1] shrink-0 ${!compact ? "text-emerald-300" : ""}`} />
    <span className="relative z-[1] max-w-full truncate">{compact ? shortLabel[href] ?? label : label}</span>
  </Link>;
}

/** Fixed, compact primary navigation that replaces the nested-scroll side rail on every screen size. */
export default function NavigationDock() {
  const pathname = usePathname();
  const menuRef = useRef<HTMLDetailsElement>(null);
  useEffect(() => { menuRef.current?.removeAttribute("open"); }, [pathname]);
  useEffect(() => {
    const close = (event: PointerEvent) => {
      if (!menuRef.current?.contains(event.target as Node)) menuRef.current?.removeAttribute("open");
    };
    const escape = (event: KeyboardEvent) => {
      if (event.key === "Escape") menuRef.current?.removeAttribute("open");
    };
    document.addEventListener("pointerdown", close);
    document.addEventListener("keydown", escape);
    return () => { document.removeEventListener("pointerdown", close); document.removeEventListener("keydown", escape); };
  }, []);
  const primary = APP_NAVIGATION.slice(0, 5);
  const secondary = APP_NAVIGATION.slice(5);
  return <nav aria-label="Workspace navigation" className="fixed inset-x-3 bottom-[max(12px,env(safe-area-inset-bottom))] z-40 mx-auto flex max-w-[650px] items-center justify-between gap-1 rounded-[23px] border border-white/[0.09] bg-[#111720]/95 p-2 shadow-[0_18px_60px_-24px_rgba(0,0,0,.95)] backdrop-blur-2xl sm:inset-x-6 sm:gap-2 sm:p-2.5">
    {primary.map(({ href, label, icon }) => <NavItem key={href} href={href} label={label} icon={icon} compact />)}
    <details ref={menuRef} className="group relative flex h-14 min-w-0 flex-1 justify-center rounded-2xl">
      <summary aria-label="More workspace sections" className="flex min-h-14 w-full cursor-pointer list-none flex-col items-center justify-center gap-1 rounded-2xl px-2 py-2 text-[10px] text-slate-500 hover:bg-white/[0.07] hover:text-slate-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400 marker:hidden"><Ellipsis size={22} strokeWidth={1.8} /><span>More</span></summary>
      <div className="absolute bottom-[calc(100%+12px)] right-0 w-60 overflow-hidden rounded-2xl border border-white/10 bg-[#17202C] p-1.5 shadow-2xl">
        {secondary.map(({ href, label, icon }) => <NavItem key={href} href={href} label={label} icon={icon} />)}
        <Link href="/privacy" className="block rounded-xl px-3 py-2.5 text-xs text-slate-400 hover:bg-white/[0.06] hover:text-white">Privacy</Link>
      </div>
    </details>
  </nav>;
}
