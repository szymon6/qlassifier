"use client";

import type { ReactNode } from "react";
import { Wordmark } from "./wordmark";

interface TopBarProps {
  tiles: ReactNode;
  metric: ReactNode;
  actions: ReactNode;
}

export function TopBar({ tiles, metric, actions }: TopBarProps) {
  return (
    <header className="flex h-[72px] shrink-0 items-center gap-10" data-testid="top-bar">
      <div className="flex items-center text-foreground">
        <Wordmark />
      </div>
      <div className="flex min-w-0 flex-1 items-center">{tiles}</div>
      <div className="flex items-center gap-6">
        {metric}
        <div className="flex items-center gap-1">{actions}</div>
      </div>
    </header>
  );
}
