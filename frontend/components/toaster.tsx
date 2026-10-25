"use client";

import { Toaster as Sonner, toast } from "sonner";
import type { ApiErrorBody } from "@/lib/types";

/** Error toast: the API's message (bold) and hint, verbatim (US-5, TDD §6.5). */
export function showError(error: Pick<ApiErrorBody, "message" | "hint" | "code">): void {
  toast.custom(
    () => (
      <div
        role="status"
        aria-atomic="true"
        data-testid="toast"
        data-code={error.code}
        className="card w-[360px] px-5 py-4"
      >
        <p
          data-testid="toast-message"
          className="text-[14px] leading-5 font-semibold text-foreground"
        >
          {error.message}
        </p>
        {error.hint ? (
          <p data-testid="toast-hint" className="mt-1 text-[13px] leading-5 text-muted-foreground">
            {error.hint}
          </p>
        ) : null}
      </div>
    ),
    { id: `${error.code}:${error.message}`, duration: 8000 },
  );
}

export function Toaster() {
  return (
    <Sonner
      position="bottom-right"
      offset={40}
      gap={12}
      visibleToasts={3}
      containerAriaLabel="Notifications"
    />
  );
}
