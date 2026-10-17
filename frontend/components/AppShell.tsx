"use client";

import type { ReactNode } from "react";
import { MotionConfig } from "framer-motion";
import PageTransition from "@/components/PageTransition";
import QuantHeader from "@/components/QuantHeader";
import { usePortfolios } from "@/components/PortfolioProvider";
import NavigationDock from "@/components/NavigationDock";

export default function AppShell({ children }: { children: ReactNode }) {
  const { activePortfolio } = usePortfolios();
  const portfolioKey = activePortfolio ? `${activePortfolio.owner_id}:${activePortfolio.id}:${activePortfolio.updated_at}` : "no-portfolio";
  return (
    <MotionConfig reducedMotion="user">
    <div className="flex min-h-screen gap-2 p-2 pb-24 sm:gap-3 sm:p-3 sm:pb-24">
      <div className="flex min-w-0 flex-1 flex-col">
        <QuantHeader />
        <main className="w-full min-w-0 flex-1 px-3 py-6 sm:px-5 sm:py-7 xl:px-8 xl:py-9 2xl:px-10">
          <PageTransition key={portfolioKey}>{children}</PageTransition>
        </main>
      </div>
      <NavigationDock />
    </div>
    </MotionConfig>
  );
}
