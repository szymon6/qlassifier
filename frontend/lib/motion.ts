// Motion: 240 ms ease-out on data changes only; none with prefers-reduced-motion (TDD §8.5).
"use client";

import { easeCubicOut } from "d3-ease";
import { timer } from "d3-timer";
import { useEffect, useRef, useState, useSyncExternalStore } from "react";

export const DURATION_MS = 240;
const QUERY = "(prefers-reduced-motion: reduce)";

function subscribeReduced(callback: () => void): () => void {
  if (typeof window === "undefined" || !window.matchMedia) return () => undefined;
  const media = window.matchMedia(QUERY);
  media.addEventListener("change", callback);
  return () => media.removeEventListener("change", callback);
}

function readReduced(): boolean {
  if (typeof window === "undefined" || !window.matchMedia) return false;
  return window.matchMedia(QUERY).matches;
}

export function useReducedMotion(): boolean {
  return useSyncExternalStore(subscribeReduced, readReduced, () => false);
}

/**
 * Interpolates plain data: finite numbers tween, arrays of equal length and objects tween
 * key by key, anything else (null, new keys, strings) snaps to the target at once.
 */
export function lerp<T>(from: unknown, to: T): (t: number) => T {
  if (typeof to === "number") {
    if (typeof from === "number" && Number.isFinite(from) && Number.isFinite(to)) {
      return ((t: number) => from + (to - from) * t) as (t: number) => T;
    }
    return () => to;
  }
  if (Array.isArray(to)) {
    if (Array.isArray(from) && from.length === to.length) {
      const parts = to.map((v, i) => lerp(from[i], v));
      return ((t: number) => parts.map((f) => f(t))) as (t: number) => T;
    }
    return () => to;
  }
  if (to && typeof to === "object") {
    if (from && typeof from === "object" && !Array.isArray(from)) {
      const source = from as Record<string, unknown>;
      const entries = Object.entries(to as Record<string, unknown>).map(
        ([k, v]) => [k, k in source ? lerp(source[k], v) : () => v] as const,
      );
      return ((t: number) => {
        const out: Record<string, unknown> = {};
        for (const [k, f] of entries) out[k] = f(t);
        return out;
      }) as (t: number) => T;
    }
    return () => to;
  }
  return () => to;
}

let running = 0;

function markTweening(delta: number): void {
  running = Math.max(0, running + delta);
  if (typeof document === "undefined") return;
  if (running > 0) document.documentElement.dataset.tweening = "true";
  else delete document.documentElement.dataset.tweening;
}

/**
 * Tweens `target` (plain data) whenever `key` changes. `key` must change exactly when the
 * target's content changes (e.g. JSON of the target).
 */
export function useTween<T extends object>(
  target: T,
  key: string,
  enabled: boolean,
  duration = DURATION_MS,
): T {
  const [snapshot, setSnapshot] = useState<{ key: string; target: T }>({ key, target });
  if (snapshot.key !== key) setSnapshot({ key, target });
  const [value, setValue] = useState<T>(target);
  const current = useRef<T>(target);

  useEffect(() => {
    const to = snapshot.target;
    if (!enabled || current.current === to) {
      current.current = to;
      return;
    }
    const step = lerp<T>(current.current, to);
    markTweening(1);
    let finished = false;
    const t = timer((elapsed) => {
      const k = Math.min(1, elapsed / duration);
      const next = k >= 1 ? to : step(easeCubicOut(k));
      current.current = next;
      setValue(next);
      if (k >= 1) {
        t.stop();
        finished = true;
        markTweening(-1);
      }
    });
    return () => {
      t.stop();
      if (!finished) markTweening(-1);
    };
  }, [snapshot, enabled, duration]);

  return enabled ? value : target;
}
