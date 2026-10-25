"use client";

import { Moon, Sun } from "lucide-react";
import { useSyncExternalStore } from "react";
const ICON_BUTTON =
  "grid size-10 place-items-center rounded-full text-muted-foreground transition-[background-color,color] duration-200 hover:bg-band hover:text-foreground";

const KEY = "qlassifier-theme";
const listeners = new Set<() => void>();

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

function isDark(): boolean {
  return document.documentElement.classList.contains("dark");
}

export function setTheme(dark: boolean): void {
  document.documentElement.classList.toggle("dark", dark);
  try {
    localStorage.setItem(KEY, dark ? "dark" : "light");
  } catch {
    // Storage blocked: the choice lasts for this page only.
  }
  listeners.forEach((l) => l());
}

/** Control 5: light / dark; starts from prefers-color-scheme (set before paint in layout). */
export function ThemeToggle() {
  const dark = useSyncExternalStore(subscribe, isDark, () => false);
  return (
    <button
      type="button"
      aria-label="Dark theme"
      aria-pressed={dark}
      data-testid="theme"
      className={ICON_BUTTON}
      onClick={() => setTheme(!dark)}
    >
      {dark ? (
        <Sun className="size-[18px]" strokeWidth={2} />
      ) : (
        <Moon className="size-[18px]" strokeWidth={2} />
      )}
    </button>
  );
}
