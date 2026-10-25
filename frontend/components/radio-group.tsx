"use client";

import { cn } from "cn";
import { useRef, type KeyboardEvent, type ReactNode } from "react";

export interface RadioOption<T extends string> {
  value: T;
  /** Accessible name. */
  label: string;
  /** Hover definition (native tooltip), e.g. what an abbreviation stands for. */
  title?: string;
  content: ReactNode;
  className?: string;
}

interface RadioGroupProps<T extends string> {
  label: string;
  value: T | null;
  options: RadioOption<T>[];
  onChange: (value: T) => void;
  className?: string;
  itemClassName?: string;
}

/**
 * WAI-ARIA radio group: one tab stop; arrows move and select; Enter / Space select.
 * Used for the dataset tiles and the metric switch (TDD §8.2, §8.6).
 */
export function RadioGroup<T extends string>({
  label,
  value,
  options,
  onChange,
  className,
  itemClassName,
}: RadioGroupProps<T>) {
  const refs = useRef<(HTMLButtonElement | null)[]>([]);
  const selected = options.findIndex((o) => o.value === value);
  const tabStop = selected === -1 ? 0 : selected;

  const move = (from: number, delta: number) => {
    const next = (from + delta + options.length) % options.length;
    onChange(options[next].value);
    refs.current[next]?.focus();
  };

  const onKeyDown = (event: KeyboardEvent<HTMLButtonElement>, index: number) => {
    switch (event.key) {
      case "ArrowRight":
      case "ArrowDown":
        event.preventDefault();
        move(index, 1);
        break;
      case "ArrowLeft":
      case "ArrowUp":
        event.preventDefault();
        move(index, -1);
        break;
      case "Home":
        event.preventDefault();
        move(0, 0);
        break;
      case "End":
        event.preventDefault();
        move(options.length - 1, 0);
        break;
      case " ":
      case "Enter":
        event.preventDefault();
        onChange(options[index].value);
        break;
      default:
        break;
    }
  };

  return (
    <div role="radiogroup" aria-label={label} className={className}>
      {options.map((option, index) => {
        const checked = option.value === value;
        return (
          <button
            key={option.value}
            ref={(el) => {
              refs.current[index] = el;
            }}
            type="button"
            role="radio"
            aria-checked={checked}
            aria-label={option.label}
            title={option.title}
            tabIndex={index === tabStop ? 0 : -1}
            data-state={checked ? "on" : "off"}
            onClick={() => onChange(option.value)}
            onKeyDown={(event) => onKeyDown(event, index)}
            className={cn(itemClassName, option.className)}
          >
            {option.content}
          </button>
        );
      })}
    </div>
  );
}
