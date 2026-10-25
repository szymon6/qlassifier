import { describe, expect, it, vi } from "vitest";
import { streamOf } from "./helpers";
import { SseParser, subscribe, type SseMessage } from "@/lib/events";

function parse(chunks: string[]): SseMessage[] {
  const out: SseMessage[] = [];
  const parser = new SseParser((m) => out.push(m));
  for (const c of chunks) parser.push(c);
  return out;
}

const sse = (id: number, event: string, data: unknown) =>
  `id: ${id}\nevent: ${event}\ndata: ${JSON.stringify(data)}\n\n`;

function response(chunks: string[], status = 200): Response {
  return new Response(streamOf(chunks), {
    status,
    headers: { "Content-Type": "text/event-stream" },
  });
}

describe("SseParser", () => {
  it("parses events split across chunks", () => {
    const text = sse(1, "progress", { done: 1 }) + sse(2, "partial", { a: [1, 2] });
    const chunks = [text.slice(0, 7), text.slice(7, 30), text.slice(30)];
    expect(parse(chunks)).toEqual([
      { id: "1", event: "progress", data: '{"done":1}' },
      { id: "2", event: "partial", data: '{"a":[1,2]}' },
    ]);
  });

  it("handles CRLF and CR line ends, comments, multi-line data and default type", () => {
    const out = parse([": ping\r\n\r\n", "data: a\r", "\ndata: b\r\n\r\n", "id: 7\rdata:c\r\r"]);
    expect(out).toEqual([
      { id: null, event: "message", data: "a\nb" },
      { id: "7", event: "message", data: "c" },
    ]);
  });

  it("ignores events without data and ids with NUL", () => {
    expect(parse(["event: x\n\n", "id: 1\u0000\ndata: y\n\n", "field-only\n\n"])).toEqual([
      { id: null, event: "message", data: "y" },
    ]);
  });
});

describe("subscribe", () => {
  it("delivers events with numeric ids and stops at a terminal event", async () => {
    const fetchImpl = vi.fn(async () =>
      response([
        sse(1, "progress", { done: 1 }),
        sse(2, "complete", { result_url: "/x" }),
        sse(3, "progress", {}),
      ]),
    );
    const got: [string, number | null][] = [];
    const sub = subscribe("/events", { fetchImpl, onMessage: (m, id) => got.push([m.event, id]) });
    await sub.closed;
    expect(got).toEqual([
      ["progress", 1],
      ["complete", 2],
    ]);
    expect(fetchImpl).toHaveBeenCalledTimes(1);
    const init = fetchImpl.mock.calls[0] as unknown as [string, RequestInit];
    expect((init[1].headers as Record<string, string>)["Last-Event-ID"]).toBeUndefined();
  });

  it("reconnects with Last-Event-ID when the stream drops before the end", async () => {
    const calls: Record<string, string>[] = [];
    const bodies = [
      [sse(1, "progress", {}), sse(2, "partial", {})],
      [sse(3, "progress", {}), sse(4, "cancelled", {})],
    ];
    const fetchImpl = vi.fn(async (_url: string, init?: RequestInit) => {
      calls.push({ ...(init?.headers as Record<string, string>) });
      return response(bodies[calls.length - 1]);
    });
    const ids: (number | null)[] = [];
    const sleep = vi.fn(async () => undefined);
    const sub = subscribe("/events", { fetchImpl, sleep, onMessage: (_m, id) => ids.push(id) });
    await sub.closed;
    expect(ids).toEqual([1, 2, 3, 4]);
    expect(calls[0]["Last-Event-ID"]).toBeUndefined();
    expect(calls[1]["Last-Event-ID"]).toBe("2");
    expect(sleep).toHaveBeenCalledTimes(1);
  });

  it("resumes from a given id", async () => {
    const fetchImpl = vi.fn(async () => response([sse(10, "failed", { error: {} })]));
    const sub = subscribe("/events", { fetchImpl, lastEventId: 9, onMessage: () => undefined });
    await sub.closed;
    const init = fetchImpl.mock.calls[0] as unknown as [string, RequestInit];
    expect((init[1].headers as Record<string, string>)["Last-Event-ID"]).toBe("9");
  });

  it("reports an HTTP error once and does not retry", async () => {
    const body = {
      error: {
        code: "run_not_found",
        message: "No run 'r-9' in this session.",
        hint: "Start a new run.",
        status: 404,
      },
    };
    const fetchImpl = vi.fn(async () => new Response(JSON.stringify(body), { status: 404 }));
    const onError = vi.fn();
    const sub = subscribe("/events", { fetchImpl, onMessage: () => undefined, onError });
    await sub.closed;
    expect(fetchImpl).toHaveBeenCalledTimes(1);
    expect(onError).toHaveBeenCalledTimes(1);
    expect(onError.mock.calls[0][0].message).toBe("No run 'r-9' in this session.");
    expect(onError.mock.calls[0][0].hint).toBe("Start a new run.");
  });

  it("gives up after the retries and reports the lost server", async () => {
    const fetchImpl = vi.fn(async () => {
      throw new TypeError("Failed to fetch");
    });
    const onLost = vi.fn();
    const sleep = vi.fn(async () => undefined);
    const sub = subscribe("/events", {
      fetchImpl,
      sleep,
      maxRetries: 2,
      onMessage: () => undefined,
      onLost,
    });
    await sub.closed;
    expect(fetchImpl).toHaveBeenCalledTimes(3);
    expect(onLost).toHaveBeenCalledTimes(1);
    expect(onLost.mock.calls[0][0].message).toBe("Server stopped.");
  });

  it("stops quietly when closed", async () => {
    let release: () => void = () => undefined;
    const fetchImpl = vi.fn(
      (_url: string, init?: RequestInit) =>
        new Promise<Response>((_resolve, reject) => {
          release = () => reject(new DOMException("aborted", "AbortError"));
          init?.signal?.addEventListener("abort", () => release());
        }),
    );
    const onLost = vi.fn();
    const onMessage = vi.fn();
    const sub = subscribe("/events", { fetchImpl, onMessage, onLost });
    sub.close();
    await sub.closed;
    expect(onLost).not.toHaveBeenCalled();
    expect(onMessage).not.toHaveBeenCalled();
  });

  it("treats a missing body as a dropped stream", async () => {
    let n = 0;
    const fetchImpl = vi.fn(async () => {
      n += 1;
      return n === 1 ? new Response(null, { status: 200 }) : response([sse(1, "complete", {})]);
    });
    const sleep = vi.fn(async () => undefined);
    const got: string[] = [];
    const sub = subscribe("/events", { fetchImpl, sleep, onMessage: (m) => got.push(m.event) });
    await sub.closed;
    expect(got).toEqual(["complete"]);
  });
});
