// Contract (TDD §6.7): the generated TypeScript types match the committed JSON Schemas, and
// the payload shapes the dashboard relies on validate against those schemas.
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import Ajv2020 from "ajv/dist/2020";
import { describe, expect, it } from "vitest";
import { makeResult } from "./helpers";

const ROOT = resolve(__dirname, "..");
const SCHEMAS = resolve(ROOT, "../src/qlassifier/schemas");
const schema = (name: string) => JSON.parse(readFileSync(resolve(SCHEMAS, `${name}.json`), "utf8"));

function validator(name: string) {
  const ajv = new Ajv2020({ strict: false, allErrors: true });
  return ajv.compile(schema(name));
}

describe("contract", () => {
  it("lib/api-types.gen.ts is in sync with src/qlassifier/schemas/api.json", () => {
    const out = execFileSync(process.execPath, ["scripts/gen-types.mjs", "--check"], {
      cwd: ROOT,
      encoding: "utf8",
    });
    expect(out).toContain("in sync");
  });

  it("every contract model is exported once as a TypeScript type", () => {
    const text = readFileSync(resolve(ROOT, "lib/api-types.gen.ts"), "utf8");
    const defs = Object.keys(schema("api").$defs as Record<string, unknown>);
    for (const name of defs) {
      expect(text, name).toMatch(new RegExp(`export (interface|type) ${name}\\b`));
    }
  });

  it("the synthetic test results are valid BenchmarkResults (helpers stay honest)", () => {
    const validate = validator("benchmark-result");
    const ok = validate(makeResult([{ id: "qlassifier", values: [0.9, 0.8] }]));
    expect(validate.errors ?? []).toEqual([]);
    expect(ok).toBe(true);
  });

  it("the error body the toasts read matches the schema", () => {
    const validate = validator("error-response");
    const body = {
      error: {
        code: "unknown_dataset",
        message: "No dataset 'iris'.",
        hint: "Choose one of: breast_cancer, ionosphere, sonar, heart, protein_angles.",
        status: 404,
      },
    };
    expect(validate(body)).toBe(true);
    expect(validate({ error: { code: "x" } })).toBe(false);
  });
});
