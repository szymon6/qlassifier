"use client";

import { useCallback, useState } from "react";

export interface Size {
  width: number;
  height: number;
}

/** Callback ref + content-box size via ResizeObserver (0 × 0 until measured). */
export function useSize<T extends HTMLElement>(): [(node: T | null) => void, Size] {
  const [size, setSize] = useState<Size>({ width: 0, height: 0 });
  const ref = useCallback((node: T | null) => {
    if (!node) return;
    const update = (width: number, height: number) =>
      setSize((prev) =>
        Math.abs(prev.width - width) < 0.5 && Math.abs(prev.height - height) < 0.5
          ? prev
          : { width, height },
      );
    const rect = node.getBoundingClientRect();
    update(rect.width, rect.height);
    if (typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver((entries) => {
      const box = entries[0]?.contentRect;
      if (box) update(box.width, box.height);
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, []);
  return [ref, size];
}
