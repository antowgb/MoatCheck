"use client";

import { motion, useReducedMotion } from "framer-motion";

export default function Card({
  title,
  children,
  className = "",
  id,
  delay = 0,
}: {
  title?: string;
  children: React.ReactNode;
  className?: string;
  id?: string;
  delay?: number;
}) {
  const reduceMotion = useReducedMotion();
  return (
    <motion.div
      id={id}
      initial={reduceMotion ? false : { opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      transition={reduceMotion ? { duration: 0 } : { duration: 0.42, delay, ease: "easeOut" }}
      className={`bento-panel p-5 transition-colors sm:p-6 ${className}`}
    >
      {title && (
        <h2 className="mb-4 text-xs font-semibold uppercase tracking-[0.14em] text-slate-400">
          {title}
        </h2>
      )}
      {children}
    </motion.div>
  );
}
