import type { Metadata, Viewport } from "next";
import localFont from "next/font/local";
import { paletteCss } from "@/lib/palette";
import "./globals.css";

// Inter Variable, self-hosted from @fontsource-variable/inter (latin subset). No CDN (NFR-2).
const inter = localFont({
  src: "../node_modules/@fontsource-variable/inter/files/inter-latin-wght-normal.woff2",
  weight: "100 900",
  style: "normal",
  display: "swap",
  variable: "--font-inter",
});

export const metadata: Metadata = {
  title: "Qlassifier",
  description: "Qlassifier vs every baseline, live.",
  icons: { icon: "/icon.svg" },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
};

// Theme before first paint: saved choice, else prefers-color-scheme (TDD §8.2).
const THEME_SCRIPT = `(function(){try{var t=localStorage.getItem("qlassifier-theme");var d=t?t==="dark":window.matchMedia("(prefers-color-scheme: dark)").matches;document.documentElement.classList.toggle("dark",d);}catch(e){}})();`;

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={inter.variable} suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_SCRIPT }} />
        <style dangerouslySetInnerHTML={{ __html: paletteCss() }} />
      </head>
      <body>{children}</body>
    </html>
  );
}
