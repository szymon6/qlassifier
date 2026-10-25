import type { Chip } from "@/lib/chips";
import { TERMS } from "@/lib/glossary";

const LABEL_TITLES: Record<Chip["id"], string> = {
  zz: TERMS.vsZz,
  noise: TERMS.noise,
  cost: TERMS.cost,
};

/** Splits "+37.6 pp" / "1,420×" into number and unit for typography. */
function split(value: string): [string, string] {
  const match = value.match(/^(.*?)( ?pp|×)$/);
  if (!match) return [value, ""];
  return [match[1], match[2].trim()];
}

/** Proof chips (TDD §8.3): three numbers; hover shows CI / source. Not controls. */
export function ProofChips({ chips }: { chips: Chip[] }) {
  return (
    <section
      aria-label="Proof"
      className="flex h-[112px] shrink-0 items-center gap-20 pl-1"
      data-testid="chips"
    >
      {chips.map((chip) => {
        const [number, unit] = split(chip.value);
        const detail = chip.detail.filter((d) => d.value !== "—");
        return (
          <div
            key={chip.id}
            className="group relative"
            data-testid={`chip-${chip.id}`}
            data-value={chip.raw ?? ""}
          >
            <p className="flex items-baseline gap-1.5 text-[44px] leading-[48px] font-semibold tracking-[-0.035em] text-foreground">
              <span className={chip.raw == null ? "text-muted-foreground opacity-40" : undefined}>
                {chip.raw == null ? "\u2013" : number}
              </span>
              {unit ? (
                <span
                  className="text-[22px] font-medium tracking-[-0.01em] text-muted-foreground"
                  title={unit === "pp" ? TERMS.pp : TERMS.ratio}
                  aria-label={unit === "pp" ? TERMS.pp : TERMS.ratio}
                >
                  {unit}
                </span>
              ) : null}
            </p>
            <p
              className="label mt-1.5"
              title={LABEL_TITLES[chip.id]}
              aria-label={LABEL_TITLES[chip.id]}
            >
              {chip.label}
            </p>
            {detail.length ? (
              <>
                <dl className="sr-only">
                  {detail.map((d) => (
                    <div key={d.label}>
                      <dt>{d.label}</dt>
                      <dd>{d.value}</dd>
                    </div>
                  ))}
                </dl>
                <div
                  aria-hidden="true"
                  className="pointer-events-none absolute top-full left-0 z-30 mt-3 translate-y-1 opacity-0 transition-[opacity,transform] duration-150 group-hover:translate-y-0 group-hover:opacity-100"
                >
                  <dl className="card grid grid-cols-[auto_auto] gap-x-5 gap-y-1 px-4 py-3 text-[12px] leading-[18px] whitespace-nowrap">
                    {detail.map((d) => (
                      <div key={d.label} className="contents">
                        <dt className="text-muted-foreground">{d.label}</dt>
                        <dd className="text-right font-medium text-foreground">{d.value}</dd>
                      </div>
                    ))}
                  </dl>
                </div>
              </>
            ) : null}
          </div>
        );
      })}
    </section>
  );
}
