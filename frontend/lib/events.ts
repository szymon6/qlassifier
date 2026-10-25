// Server-Sent Events client for run progress (TDD §6.4, ADR 0007).
// A fetch-stream reader instead of the browser EventSource: it sends `Last-Event-ID` on every
// reconnect (also after a reload), stops on terminal events and on HTTP errors, and reports
// a lost server once instead of retrying forever.
import { ApiError, SERVER_STOPPED, errorFromResponse } from "./api";

export interface SseMessage {
  id: string | null;
  event: string;
  data: string;
}

/** Incremental parser for the `text/event-stream` format (WHATWG HTML §9.2.6). */
export class SseParser {
  private buffer = "";
  private data: string[] = [];
  private event = "";
  private id: string | null = null;
  private sawCr = false;

  constructor(private readonly emit: (message: SseMessage) => void) {}

  push(chunk: string): void {
    let text = chunk;
    if (this.sawCr && text.startsWith("\n")) text = text.slice(1);
    this.sawCr = false;
    this.buffer += text;
    let start = 0;
    for (let i = 0; i < this.buffer.length; i++) {
      const ch = this.buffer[i];
      if (ch !== "\n" && ch !== "\r") continue;
      this.line(this.buffer.slice(start, i));
      if (ch === "\r") {
        if (i + 1 < this.buffer.length) {
          if (this.buffer[i + 1] === "\n") i++;
        } else {
          this.sawCr = true;
        }
      }
      start = i + 1;
    }
    this.buffer = this.buffer.slice(start);
  }

  private line(line: string): void {
    if (line === "") {
      this.dispatch();
      return;
    }
    if (line.startsWith(":")) return;
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "data") this.data.push(value);
    else if (field === "event") this.event = value;
    else if (field === "id" && !value.includes("\u0000")) this.id = value;
  }

  private dispatch(): void {
    if (this.data.length === 0) {
      this.event = "";
      return;
    }
    const message: SseMessage = {
      id: this.id,
      event: this.event || "message",
      data: this.data.join("\n"),
    };
    this.data = [];
    this.event = "";
    this.emit(message);
  }
}

export const TERMINAL_EVENTS = new Set(["complete", "cancelled", "failed"]);

type FetchLike = (input: string, init?: RequestInit) => Promise<Response>;

export interface SubscribeOptions {
  /** Resume after this event id (sent as `Last-Event-ID`). */
  lastEventId?: number | null;
  onMessage: (message: SseMessage, eventId: number | null) => void;
  /** HTTP error from the stream endpoint (e.g. 404 run_not_found). No retry. */
  onError?: (error: ApiError) => void;
  /** The server could not be reached after `maxRetries` attempts. */
  onLost?: (error: ApiError) => void;
  fetchImpl?: FetchLike;
  retryMs?: number;
  maxRetries?: number;
  sleep?: (ms: number) => Promise<void>;
}

export interface Subscription {
  close(): void;
  readonly closed: Promise<void>;
}

const defaultSleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

export function subscribe(url: string, options: SubscribeOptions): Subscription {
  const doFetch: FetchLike = options.fetchImpl ?? ((input, init) => fetch(input, init));
  const sleep = options.sleep ?? defaultSleep;
  const retryMs = options.retryMs ?? 1000;
  const maxRetries = options.maxRetries ?? 5;
  const controller = new AbortController();
  let lastEventId = options.lastEventId ?? null;
  let stopped = false;

  const run = async () => {
    let failures = 0;
    while (!stopped) {
      let terminal = false;
      try {
        const headers: Record<string, string> = { Accept: "text/event-stream" };
        if (lastEventId != null) headers["Last-Event-ID"] = String(lastEventId);
        const response = await doFetch(url, {
          headers,
          signal: controller.signal,
          cache: "no-store",
        });
        if (!response.ok) {
          const error = await errorFromResponse(response);
          if (!stopped) options.onError?.(error);
          return;
        }
        failures = 0;
        const body = response.body;
        if (!body) throw new Error("empty stream");
        const reader = body.getReader();
        const decoder = new TextDecoder();
        const parser = new SseParser((message) => {
          if (stopped) return;
          const numeric =
            message.id != null && /^\d+$/.test(message.id) ? Number(message.id) : null;
          if (numeric != null) lastEventId = numeric;
          if (TERMINAL_EVENTS.has(message.event)) terminal = true;
          options.onMessage(message, numeric);
        });
        while (!stopped && !terminal) {
          const { value, done } = await reader.read();
          if (done) break;
          parser.push(decoder.decode(value, { stream: true }));
        }
        if (terminal || stopped) {
          await reader.cancel().catch(() => undefined);
          return;
        }
        // Stream ended without a terminal event: reconnect and resume.
      } catch {
        if (stopped) return;
        failures += 1;
        if (failures > maxRetries) {
          options.onLost?.(new ApiError(SERVER_STOPPED));
          return;
        }
      }
      if (stopped) return;
      await sleep(retryMs * Math.max(1, failures));
    }
  };

  const closed = run().finally(() => {
    stopped = true;
  });

  return {
    close() {
      stopped = true;
      controller.abort();
    },
    closed,
  };
}
